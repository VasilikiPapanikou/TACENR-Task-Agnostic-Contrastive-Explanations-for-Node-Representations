
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from sklearn.utils import shuffle
from nodeFeatures import NodeFeatures
from pyHSICLasso import HSICLasso
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm 
import networkx as nx
from pygam import LinearGAM, LogisticGAM, s
import torch
 

class Explainer:

    def __init__(self, G_nx, G_pyg, node_features_df, all_pprs, node_ids, features, graph_dataset_name, embedding_method, embeddings, gnn_model, explanation_type = "local", node_attributes = False, structural_features = True,  problem="classification", model_type='logistic', num_similar=150, num_dissimilar=150, scale=True, parameter_tuning = True, enable_plots=False, contrastive=True, weighting_method = None, include_target_node_features = False, use_proximity = True):
        
        self.G_nx = G_nx
        self.G_pyg = G_pyg
        self.node_features_df = node_features_df
        self.all_pprs = all_pprs
        self.node_ids = node_ids
        self.features = features
        self.graph_dataset_name = graph_dataset_name  
        self.embedding_method = embedding_method
        self.embeddings = embeddings
        self.gnn_model = gnn_model
        self.explanation_type = explanation_type
        self.node_attributes = node_attributes
        self.structural_features = structural_features
        self.problem = problem
        self.model_type = model_type
        self.num_similar = num_similar
        self.num_dissimilar = num_dissimilar
        self.scale = scale
        self.parameter_tuning = parameter_tuning
        self.enable_plots = enable_plots
        self.contrastive = contrastive   
        self.weighting_method = weighting_method   
        self.include_target_node_features = include_target_node_features
        self.use_proximity = use_proximity
        
    def _prep_global_cache(self):

        """
        Precompute standardized, unit-normalized embeddings and per-dim global var.
        """
        
        if getattr(self, "_glob_ready", False):
            return
        Z = np.ascontiguousarray(self.embeddings, dtype=np.float64)   
        mu = Z.mean(axis=0, keepdims=True)
        sigma = Z.std(axis=0, ddof=1, keepdims=True) + 1e-12
        Zs = (Z - mu) / sigma
        Z_unit  = Z  / (np.linalg.norm(Z,  axis=1, keepdims=True) + 1e-12)
        Zs_unit = Zs / (np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-12)
        var_global_s = np.var(Z, axis=0, ddof=1)

        inverse_variance_global = 1.0 / (np.var(Z, axis=0) + 1e-12)
        self._Z = Z
        self._mu = mu
        self._sigma = sigma
        self._Zs = Zs
        self._var_global_s = var_global_s
        self.inverse_variance_global = inverse_variance_global
         
        self._Z_unit = np.ascontiguousarray(Z_unit,  dtype=np.float32)
        self._Zs_unit = np.ascontiguousarray(Zs_unit, dtype=np.float32)
        self._glob_ready = True

     
    def _fisher_snr_weights(self):

        """
        Compute Fisher SNR weights: between-class variance / within-class variance
        Higher SNR = dimension better at separating classes
       
        """
         
        embeddings = self.embeddings
        labels = self.G_pyg.y.cpu().numpy()
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        n_features = embeddings.shape[1]
        global_mean = np.mean(embeddings, axis=0)
        between_var = np.zeros(n_features)
        for label in unique_labels:
            class_mask = (labels == label)
            class_mean = np.mean(embeddings[class_mask], axis=0)
            n_class = np.sum(class_mask)
            between_var += n_class * (class_mean - global_mean) ** 2
        between_var /= (n_classes - 1)  
        within_var = np.zeros(n_features)
        for label in unique_labels:
            class_mask = (labels == label)
            class_embeddings = embeddings[class_mask]
            class_mean = np.mean(class_embeddings, axis=0)
            within_var += np.sum((class_embeddings - class_mean) ** 2, axis=0)
        within_var /= (len(labels) - n_classes)  
        fisher_snr = between_var / (within_var + 1e-12)
        
        return fisher_snr

    def _topm_cosine_neighbors(self, i, m):

        """
        Top-m neighbors by plain cosine in standardized space (uses unit vectors).
        """

        self._prep_global_cache()
        z_i = self._Z_unit[i]                       
        sims = self._Z_unit @ z_i                   
        sims[i] = -np.inf
        N = sims.size

        if m >= N - 1:
            idx = np.argsort(-sims)
        else:
            blk = np.argpartition(-sims, m)[:m]
            idx = blk[np.argsort(-sims[blk])]
        return idx
    
    def _weights_sim_vs_dissim(self, i, eps=1e-12, normalize=True):
        
        """
        Compute per-dimension weights based on the absolute difference between
        the mean embeddings of the m most similar and n most dissimilar nodes.
        """
        m = self.num_similar
        n = self.num_dissimilar
        self._prep_global_cache()
        Zs = self._Zs 

        # Find top-m and bottom-n nodes by cosine similarity
        z_i = self._Z_unit[i]
        sims = self._Z_unit @ z_i
        sims[i] = -np.inf

        sim_idx = np.argpartition(-sims, m)[:m]
        dis_idx = np.argpartition(sims, n)[:n]

        # Compute mean embedding vectors for each group
        mu_sim = np.mean(Zs[sim_idx, :], axis=0)
        mu_dis = np.mean(Zs[dis_idx, :], axis=0)

        # Ccontrastive weight
        w = np.abs(mu_sim - mu_dis)

        # Clean up and normalize
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) + eps
        if normalize:
            s = w.sum()
            if s > 0:
                w /= s

        return w
    
    def _weights_local_vs_global(self, i, m, mode='mean', eps=1e-12, normalize=True):
        self._prep_global_cache()
        nn_idx = self._topm_cosine_neighbors(i, m)

        if mode == 'var':
            local_stat = np.var(self._Zs[nn_idx, :], axis=0, ddof=1)
            global_stat = self._var_global_s
        elif mode == 'mean':
            local_stat = np.mean(self._Zs[nn_idx, :], axis=0)
            global_stat = self._mu
        else:
            raise ValueError("mode must be 'var' or 'mean'")

        w = np.abs(global_stat - local_stat)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) + eps
        if normalize:
            s = w.sum()
            if s > 0:
                w /= s
        return w
    
    def _weighted_cosine_to_all(self, i, w, use_standardized=True):
         
        eps = 1e-12
        self._prep_global_cache()
        Z = self._Zs if use_standardized else self._Z
        Z = np.asarray(Z, float)
        w = np.asarray(w, float)
        zi = Z[i]
        num = (Z * w) @ zi    
        deni = np.sqrt((w * zi * zi).sum() + eps)
        denj = np.sqrt((w * Z * Z).sum(axis=1) + eps)
        
        sims = num / (deni * denj)
        sims[i] = -np.inf   
        
        return sims
 
    
    def weighted_cosine_similarity(self, vec1, vec2, weights):
        
        weighted_dot = np.sum(weights * vec1 * vec2)
        norm_vec1 = np.sqrt(np.sum(weights * (vec1 ** 2)))
        norm_vec2 = np.sqrt(np.sum(weights * (vec2 ** 2)))
        return weighted_dot / (norm_vec1 * norm_vec2) if (norm_vec1 * norm_vec2) != 0 else 0.0
       
 


    def _pick_target_logits(self, logits, node_idx=None):
       
        if logits.dim() == 2 and logits.size(1) > 1:
            if node_idx is None:
                top = logits.argmax(dim=1, keepdim=True)          
                return logits.gather(1, top).squeeze(1)          
            else:
                 
                return logits[node_idx].max()                     
        else: 
            return logits.squeeze(-1) if node_idx is None else logits[node_idx].reshape(())
        
     
    
    def compute_embedding_importance(self, node_idx=None, normalize=True):
        """
        Compute importance scores for either all nodes or a specific node
        
        """
      
        self.gnn_model.eval()
    
        embeddings = self.gnn_model(self.G_pyg.x, self.G_pyg.edge_index, return_embeddings=True)
        embeddings = embeddings.detach().requires_grad_(True)
        logits = self.gnn_model.out(embeddings) 
        
        target = self._pick_target_logits(logits, node_idx=node_idx)
        scalar = target.sum() if node_idx is None else target
        
         
        (raw_gradients,) = torch.autograd.grad(
        scalar, embeddings, retain_graph=False, create_graph=False, allow_unused=False
        )
     
        sal = torch.relu(raw_gradients)
        if node_idx is None:
            imp = sal.mean(dim=0)     
        else:
            imp = sal[node_idx]  
         
        eps=1e-8
        
        if normalize:
            imp = imp / (imp.abs().sum() + eps)
         
        
        return imp.cpu().numpy(), raw_gradients

    

        
    def _plain_cosine_to_all(self, i, use_standardized=True):
        
        """
        Unweighted cosine from node i to all nodes (exclude self).
        """
        self._prep_global_cache()
        U = self._Zs_unit if use_standardized else self._Z_unit 
        sims = U @ U[i]                                        
        sims[i] = -np.inf
        return sims

     
    def find_similar_dissimilar_nodes(self, node, mode = "weighted", problem = "unsupervised"):
        """
        Identifies the most similar and dissimilar nodes to a given target node based on embedding similarity.

        Computes similarity using either a weighted cosine similarity or standard cosine similarity,
        depending on the selected mode. The weights can be based on the variance of embedding dimensions 
        (for unsupervised) or node-specific importance scores (for supervised).

        Args:
            node: The index of the target node.
            mode: Similarity mode, either "weighted" for weighted cosine similarity or any other value for standard cosine similarity.
            problem: The problem setting, either "unsupervised" to use variance-based weights or "supervised" to use learned importance weights.

        
        """
        
        self._prep_global_cache()
         
        if mode == "weighted":   
            if problem == "unsupervised":
                if self.weighting_method == "global_local_only_for_ranking":
                    w = self._weights_local_vs_global(node, self.num_similar, normalize=False)
                elif self.weighting_method == "global_variance_only_for_ranking" or self.weighting_method == "global_variance":
                    w =  self._var_global_s
                
                    
                s_rank = self._weighted_cosine_to_all(node, w, use_standardized=False)         
                if self.weighting_method ==  "global_variance":
                    s_report = s_rank
                else:
                    s_report = self._plain_cosine_to_all(node, use_standardized=False) 
                

            elif problem == "supervised":

                if self.weighting_method == "fisher_snr_weights" or self.weighting_method == "fisher_snr_only_for_ranking":
                    node_imp = self._fisher_snr_weights()

                elif self.weighting_method == "gradients" or self.weighting_method == "gradients_only_for_ranking":
                   
                    node_imp, _ = self.compute_embedding_importance(node_idx=node, normalize=True)
               
                s_rank = self._weighted_cosine_to_all(node, node_imp, use_standardized=False)

                if self.weighting_method == "fisher_snr_only_for_ranking" or self.weighting_method == "gradients_only_for_ranking":
                    s_report = self._plain_cosine_to_all(node, use_standardized=False)
                else:
                    s_report = s_rank                    
        else:     
            s_rank = self._plain_cosine_to_all(node, use_standardized=False)
            s_report = s_rank
            
        
        s_rank   = np.asarray(s_rank,   dtype=float)
        s_report = np.asarray(s_report, dtype=float)

        valid = np.isfinite(s_rank)
        if 0 <= node < s_rank.size:
            valid[node] = False

        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return [], []

        
        order = idx[np.argsort(s_rank[idx])]                   
        bot_idx = order[:min(self.num_dissimilar, order.size)]  
        top_idx = order[::-1][:min(self.num_similar, order.size)]  
        similar_nodes    = [(int(i), float(s_report[i])) for i in top_idx]
        dissimilar_nodes = [(int(i), float(s_report[i])) for i in bot_idx]

        return similar_nodes, dissimilar_nodes
    
    def interpret_linear_regression(self, feature_names=None):
        """
        Interprets the fitted linear regression model by analyzing the coefficients assigned to each feature.

        Constructs a DataFrame of features and their corresponding coefficient values, sorted by absolute importance.
        Optionally generates a bar plot showing the most important features based on the magnitude of their coefficients.

        """
        coefficients = self.model.coef_.flatten()
       
        feature_names = np.array(feature_names)
    
        coeff_data = pd.DataFrame({
            "Feature": feature_names,
            "Importance": coefficients
        })
        coeff_data_sorted = coeff_data.reindex(coeff_data['Importance'].abs().sort_values(ascending=False).index)
        
        if self.enable_plots:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=coeff_data_sorted, palette="coolwarm")
            plt.xlabel("Coefficient Value")
            plt.ylabel("Feature")
            
            output_dir = f'Figures_Linear_Regression/{self.graph_dataset_name}/{self.embedding_method}'
            os.makedirs(output_dir, exist_ok=True)
           
        return coeff_data
    
    
    def fit_surrogate_model(self, features_for_similar, features_for_disimilar = None, similarities = [], gam_gridsearch=True, gam_compute_delta_r2=False, return_coefficients_from = "target"):
        """
        Fits a surrogate regression model to predict similarity scores from node feature differences.

        The method supports multiple model types, including linear regression, ridge regression, lasso regression,
        and a dummy regressor for baseline comparison. Optionally performs scaling and hyperparameter tuning.
        Returns the model's performance metrics and feature importance (for linear models).

        Args:
            features_for_similar: Feature matrix for similar node pairs.
            features_for_disimilar: Feature matrix for dissimilar node pairs.
            similarities: Target similarity scores for the node pairs.

       
        """
        
        
        if self.contrastive == True:
            X = np.vstack([features_for_similar, features_for_disimilar])
        else:
            X = features_for_similar


        y = similarities

        X, y = shuffle(X, y, random_state=42)
        
         
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        col_means = np.nanmean(X_train, axis=0)
        col_stds = np.nanstd(X_train, axis=0, ddof=1)

        nonzero_var_indices = np.where(col_stds > 0)[0]
        feature_names = self.features
        
        feature_names = [feature_names[i] for i in nonzero_var_indices]
        
        X_train = X_train[:, nonzero_var_indices]
        X_test = X_test[:, nonzero_var_indices]
        col_means = col_means[nonzero_var_indices]
        col_stds = col_stds[nonzero_var_indices]

        if self.scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
                
        else:
            scaler = None
            X_train_scaled = X_train
            X_test_scaled = X_test

        if self.model_type == "linear":
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)

        elif self.model_type == "ridge":
            self.model = Ridge()
            if self.parameter_tuning:
                param_grid = {
                    'alpha': np.logspace(-6, 6, 13)
                }
                grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error')
                grid_search.fit(X_train_scaled, y_train)
                self.model = grid_search.best_estimator_

        elif self.model_type == "lasso":
            from sklearn.linear_model import LassoCV

            self.model = LassoCV(
                alphas=100,     
                cv=5,
                max_iter=100000,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)

        elif self.model_type == "hsic_lasso":
            import warnings
            import contextlib
            warnings.simplefilter("ignore")
            X_hsic = np.asarray(X_train_scaled, dtype=np.float64)
            y_hsic = np.asarray(y_train, dtype=np.float64)
            
            y_train = np.asarray(y_train, dtype=np.float64)
            y_test = np.asarray(y_test, dtype=np.float64)
            
            d = X_hsic.shape[1]
            k = min(100, max(20, int(0.1 * d)))
           
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    hsic = HSICLasso()
                    hsic.input(X_hsic, y_hsic)
                    try:
                        hsic.regression(k)
                        selected_idx = hsic.get_index()
                    except Exception as e:
                        selected_idx = None

            w = np.zeros(X_train_scaled.shape[1])
            w[selected_idx] = 1.0

            if w.sum() > 0:
                w = w / w.sum()

            importance_data = pd.DataFrame({
                "Feature": feature_names,
                "Importance": w
            })

            if selected_idx is not None and len(selected_idx) > 0:
                lin = LinearRegression()
                lin.fit(X_train_scaled[:, selected_idx], y_train)
                y_pred = lin.predict(X_test_scaled[:, selected_idx])
            else:
                y_pred = np.full_like(y_test, y_train.mean(), dtype=np.float64)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)

            return importance_data, mse, mae, r2, rmse

        y_pred = self.model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)  
     
        if self.model_type == "linear" or self.model_type == "ridge" or self.model_type == "lasso":
            importance_data = self.interpret_linear_regression(feature_names=feature_names)  

        return importance_data, mse, mae, r2, rmse
        
    
    def cosine_similarity_vec(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    
    
    
    def explain(self, mode = "weighted", problem = "unsupervised"):
        """
        Generate explanations for node representations by analyzing which structural or attribute-based features 
        contribute most to the similarity/ dissimilarity of node representations.

        Args:
            mode: Similarity mode, either "weighted" for weighted cosine.
            problem: The problem setting, either "unsupervised" to explain usnupervised nore representations or "supervised" explain supervised node representations.
        
        """
        
        nodeFeats = NodeFeatures()
       
        all_metrics = []
        all_top_features = []
        all_importances = []
        all_node_metrics = []     

        
        if self.explanation_type == "local":
            similar_nodes, dissimilar_nodes = self.find_similar_dissimilar_nodes(self.node_ids)
        
            if self.node_attributes:
                node_attr =  self.G_pyg.x.cpu().numpy()
            
            else:
                node_attr = []
            
            similar_nodes_ids = [node_id for node_id, _ in similar_nodes]
            dissimilar_nodes_ids = [node_id for node_id, _ in dissimilar_nodes]
            similarities_similar = [similarity for _, similarity in similar_nodes]
            similarities_dissimilar = [similarity for _, similarity in dissimilar_nodes]
            similarities= similarities_similar + similarities_dissimilar

            features_for_similar = Parallel(n_jobs=-1)(delayed(nodeFeats.compute_features_for_neighborhood)(self.G_nx, node_attr, node) for node in similar_nodes_ids)
            features_for_disimilar = Parallel(n_jobs=-1)(delayed(nodeFeats.compute_features_for_neighborhood)(self.G_nx, node_attr, node) for node in dissimilar_nodes_ids)
                
            importance_data, mse , mae, r2 = self.fit_surrogate_model(features_for_similar, features_for_disimilar, similarities = similarities)
            return importance_data, mse, mae, r2
        
        
   
        else:
            for node in tqdm(self.node_ids, desc='Processing nodes'):
                
                if self.node_attributes:
                    node_attr =  self.G_pyg.x.cpu().numpy()
                else:
                    node_attr = []

                if self.contrastive == True:
                    similar_nodes, dissimilar_nodes = self.find_similar_dissimilar_nodes(node, mode = mode, problem = problem)
            

                
                    if self.problem == "regression":     
                        
                        
                        similar_nodes_ids = [node_id for node_id, _ in similar_nodes]
                        dissimilar_nodes_ids = [node_id for node_id, _ in dissimilar_nodes]
                        similarities_similar = [similarity for _, similarity in similar_nodes]
                        similarities_dissimilar = [similarity for _, similarity in dissimilar_nodes]
                        similarities = similarities_similar + similarities_dissimilar
                        
                        
                        if self.structural_features == False:
                            
                            features_for_similar = [np.array(node_attr[node]) for node in similar_nodes_ids]
                            
                            features_for_dissimilar = [np.array(node_attr[node]) for node in dissimilar_nodes_ids]
                            
                            
                            importance_data, mse, mae, r2, rmse = self.fit_surrogate_model(features_for_similar, features_for_dissimilar, similarities = similarities)
                            semantic_similarities = []
                            node_features = self.node_features_df.loc[node].values
                            for similar_node in similar_nodes_ids:
                                similar_node_features = self.node_features_df.loc[similar_node].values
                                sim =  self.cosine_similarity_vec(node_features, similar_node_features)
                                semantic_similarities.append(sim)
                            avg_semantic_similarity = np.mean(semantic_similarities)


                            all_metrics.append({
                            'MSE': mse,
                            'MAE': mae,
                            'R2': r2, 
                            'RMSE': rmse,
                            'Avg_Semantic_Similarity_With_m_Similar': avg_semantic_similarity
                            })
                            
                            all_node_metrics.append({                         
                                'Node': node,
                                'm': len(similar_nodes_ids),
                                'n': len(dissimilar_nodes_ids),
                                'MSE': mse,
                                'MAE': mae,
                                'R2': r2,
                                'RMSE': rmse 
                            })
                        else:
                            if self.use_proximity:
                                extra_features_similar = []
                                for target_node in similar_nodes_ids:
                                    
                                    dist = nx.shortest_path_length(self.G_nx, source=node, target=target_node)
                                    proximity = 1.0 / (1.0 + dist)
                                    #ppr_val = self.all_pprs[node].get(target_node, 0.0)
                                    extra_features_similar.append([proximity])#, ppr_val])
                                    #path_lengths_similar.append(proximity)

                                extra_features_dissimilar = []
                                for target_node in dissimilar_nodes_ids:
                                    
                                    dist = nx.shortest_path_length(self.G_nx, source=node, target=target_node)
                                    proximity = 1.0 / (1.0 + dist)
                                    #ppr_val = self.all_pprs[node].get(target_node, 0.0)
                                    extra_features_dissimilar.append([proximity])#, ppr_val])

                                extra_features_similar = np.array(extra_features_similar)
                                extra_features_dissimilar = np.array(extra_features_dissimilar)

                            struct_feats_sim = self.node_features_df.loc[similar_nodes_ids].values
                            struct_feats_dis = self.node_features_df.loc[dissimilar_nodes_ids].values  
                            
                            
                            #############################################

                            if self.include_target_node_features == True:
                                target_feats = self.node_features_df.loc[node].values
                                target_feats_sim = np.tile(target_feats, (len(similar_nodes_ids), 1))
                                target_feats_dis = np.tile(target_feats, (len(dissimilar_nodes_ids), 1))

                                
                            
                                diff_feats_sim = target_feats_sim - struct_feats_sim
                                diff_feats_dis = target_feats_dis - struct_feats_dis

                                if self.use_proximity:
                                    X_similar = np.hstack((diff_feats_sim, extra_features_similar))
                                    X_dissimilar = np.hstack((diff_feats_dis, extra_features_dissimilar))
                                else:
                                    X_similar = diff_feats_sim
                                    X_dissimilar = diff_feats_dis
                                
                                original_feature_names = list(self.features)
                                
                                
                                base_feature_names = [n for n in original_feature_names if n not in ["Proximity", "PPR"]]

                                diff_feature_names = [n for n in base_feature_names]
                                if self.use_proximity:
                                    self.features = diff_feature_names + ["Proximity"]
                                else:
                                    self.features = diff_feature_names

                            
                                importance_data, mse, mae, r2, rmse = self.fit_surrogate_model(
                                    X_similar, 
                                    X_dissimilar, 
                                    similarities=similarities
                                )
                                
                                
                                self.features = original_feature_names

                            else:
                                
                                
                                if self.use_proximity:
                                    X_similar = np.hstack((extra_features_similar, struct_feats_sim))
                                    X_dissimilar = np.hstack((extra_features_dissimilar, struct_feats_dis))
                                else:
                                    X_similar = struct_feats_sim
                                    X_dissimilar = struct_feats_dis
                                
                                importance_data, mse, mae, r2, rmse = self.fit_surrogate_model(
                                    X_similar, 
                                    X_dissimilar, 
                                    similarities=similarities
                                )
                            
                             
                            #################################################
                            semantic_similarities = []
                            node_features = self.node_features_df.loc[node].values
                            for similar_node in similar_nodes_ids:
                                similar_node_features = self.node_features_df.loc[similar_node].values
                                sim =  self.cosine_similarity_vec(node_features, similar_node_features)
                                semantic_similarities.append(sim)
                            avg_semantic_similarity = np.mean(semantic_similarities)


                            all_metrics.append({
                            'MSE': mse,
                            'MAE': mae,
                            'R2': r2, 
                            'RMSE': rmse,
                            'Avg_Semantic_Similarity_With_m_Similar': avg_semantic_similarity
                            })
                            
                            all_node_metrics.append({                         
                                'Node': node,
                                'm': len(similar_nodes_ids),
                                'n': len(dissimilar_nodes_ids),
                                'MSE': mse,
                                'MAE': mae,
                                'R2': r2,
                                'RMSE': rmse
                            })
                else:
                    self._prep_global_cache()
                    N = self.G_pyg.num_nodes
                    k = max(1, int(0.10 * (N - 1)))
                    rng = np.random.default_rng(42 + int(node))
                    all_ids = np.arange(N, dtype=int)
                    candidates = all_ids[all_ids != node]
                    sample_ids = rng.choice(candidates, size=min(k, candidates.size), replace=False)
                    if mode == "weighted":
                        if problem == "unsupervised":
                            w = self._var_global_s.copy()
                            sims_all = self._weighted_cosine_to_all(node, w, use_standardized=True)
                        else:  
                            node_imp, _ = self.compute_embedding_importance(node_idx=node, normalize=True)
                            sims_all = self._weighted_cosine_to_all(node, node_imp, use_standardized=False)
                    else:
                        sims_all = self._plain_cosine_to_all(node, use_standardized=False)

                     
                     
                    similarities = sims_all[sample_ids]
                    prox = []
                    for j in sample_ids:
                        try:    dist = nx.shortest_path_length(self.G_nx, source=node, target=int(j))
                        except nx.NetworkXNoPath: dist = None
                        prox.append(0.0 if dist is None else 1.0/(1.0+dist))
                    proximities = np.array(prox, dtype=np.float32).reshape(-1,1)
                    X_struct = self.node_features_df.loc[sample_ids].values
                    X_sample = np.hstack([proximities, X_struct])
                
                    self.features = ["Proximity"] + list(self.node_features_df.columns)

                    importance_data, mse, mae, r2, rmse = self.fit_surrogate_model(np.array(X_sample), similarities = similarities)
                    all_metrics.append({
                            'MSE': mse,
                            'MAE': mae,
                            'R2': r2, 
                            'RMSE': rmse
                            })
                            
                    all_node_metrics.append({                         
                        'Node': node,
                        'MSE': mse,
                        'MAE': mae,
                        'R2': r2,
                        'RMSE': rmse
                    })
                     
                importance_data_with_node = importance_data.copy()
                importance_data_with_node['Node'] = node
                all_importances.append(importance_data_with_node)

                
                top_features = importance_data.nlargest(3, 'Importance')['Feature'].tolist()
                
                all_top_features.append(top_features)

                
            frequency_count = {}
            for node in all_top_features:
                for feature in node:
                    if feature in frequency_count:
                        frequency_count[feature] += 1
                    else:
                        frequency_count[feature] = 1

            frequency_df = pd.DataFrame(list(frequency_count.items()), columns=['Feature', 'Frequency'])
            frequency_df = frequency_df.sort_values(by='Frequency', ascending=False)
            
        combined_importances = pd.concat(all_importances, axis=0)
         
        
        avg_importances = combined_importances.groupby('Feature')['Importance'].mean().reset_index()
        avg_importances_df = avg_importances.sort_values(by='Importance', ascending=False)

        average_mse = np.mean([m['MSE'] for m in all_metrics])
        average_mae = np.mean([m['MAE'] for m in all_metrics])
        average_r2 = np.mean([m['R2'] for m in all_metrics])
        average_rmse = np.mean([m['RMSE'] for m in all_metrics])
       
        all_importances_df = pd.concat(all_importances, ignore_index=True)
        
    
        all_importances_df = all_importances_df[['Node', 'Feature', 'Importance']]
        
        
        per_node_metrics_df = pd.DataFrame(all_node_metrics).sort_values('Node').reset_index(drop=True)
         

        return frequency_df, avg_importances_df, all_importances_df, {
                'MSE': average_mse,
                'MAE': average_mae,
                'R2': average_r2,
                'average_rmse': average_rmse,
                'Avg_Semantic_Similarity_With_m_Similar': avg_semantic_similarity
        },per_node_metrics_df 

        
            
           
                
