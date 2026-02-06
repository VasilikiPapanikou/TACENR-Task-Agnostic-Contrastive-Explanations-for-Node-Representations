import os.path as osp
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import networkx as nx
import torch
from torch_geometric.datasets import KarateClub
import numpy as np
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets import DeezerEurope
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import AttributedGraphDataset
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_networkx

class GraphLoader:
    """
    A class for loading graphs in various formats and converting 
    between NetworkX and PyTorch Geometric (PyG) representations.
    Supports multiple standard datasets including Facebook, Cora, Blog, Flickr, and Karate Club.
    """

    def __init__(self, graph_file_path, dataset_name):
        """
        Initializes the graph loader with the dataset name and file path.

        Args:
            graph_file_path: Path to the input graph file.
            dataset_name: Name of the dataset to load.
        """
        self.graph_file_path = graph_file_path
        self.dataset_name = dataset_name


    def nx_to_pyg(self, G_nx):
        """
        Converts a NetworkX graph to a PyTorch Geometric Data object.

        Args:
            G_nx: NetworkX graph object.
        
        """
        edge_index = torch.tensor(list(G_nx.edges), dtype=torch.long).t().contiguous()
        num_nodes = G_nx.number_of_nodes()
        x = torch.ones((num_nodes, 1))  
        is_directed = G_nx.is_directed()
        return Data(edge_index=edge_index, x=x)
    
    def pyg_to_nx(self, data):
        """
        Converts a PyTorch Geometric Data object to a NetworkX graph.

        Args:
            data: PyTorch Geometric Data object.
        
        """
        edge_index = data.edge_index

        is_directed = edge_index.shape[1] != edge_index.shape[0]
        G_nx = to_networkx(data, to_undirected=is_directed)
    
        return G_nx
    
    
    def load_graph(self):
        """
        Loads the graph based on the dataset name and returns both NetworkX and PyG versions.

        Args:
            None directly. Uses self.dataset_name and self.graph_file_path.
        
        """
      
        
        if self.dataset_name == "Cora" or self.dataset_name == "CiteSeer" or self.dataset_name == "Pubmed":
            path = osp.join('./', 'data', self.dataset_name)

            transform = T.Compose([
                T.LargestConnectedComponents(),   
                T.NormalizeFeatures()            
            ]) 
            dataset = Planetoid(path, self.dataset_name, transform=transform)
            G_pyg = dataset[0]
            G_nx = self.pyg_to_nx(G_pyg)

        elif self.dataset_name.startswith("ogbn"):
            dataset = PygNodePropPredDataset(
                name=self.dataset_name,
                root=f"data/ogb"
            )

            G_pyg = dataset[0]
            G_pyg.name = self.dataset_name
            G_nx = self.pyg_to_nx(G_pyg)
            


        elif self.dataset_name == "BAShapes":
            dataset = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=300, num_edges=5),
            motif_generator='house',
            num_motifs=80,
            )
            G_pyg = dataset[0]
            G_nx = self.pyg_to_nx(G_pyg)

        elif self.dataset_name == "DeezerEurope":
            dataset = DeezerEurope(root="data/DeezerEurope", transform=NormalizeFeatures())
            G_pyg = dataset[0]
            print(G_pyg)
            G_nx = self.pyg_to_nx(G_pyg)

        elif self.dataset_name == "Yelp":
            from torch_geometric.datasets import Yelp

            dataset = Yelp(root="data/Yelp", transform=NormalizeFeatures())
            G_pyg = dataset[0]
            G_nx = self.pyg_to_nx(G_pyg)
        
        elif self.dataset_name == "Twitch":
            from torch_geometric.datasets import Twitch

            dataset = Twitch(root="data/Twitch", name="EN", transform=NormalizeFeatures())
           
            G_pyg = dataset[0]
            G_nx = self.pyg_to_nx(G_pyg)
        
        elif self.dataset_name == "Reddit2":
            from torch_geometric.datasets import Reddit2

            dataset = Reddit2(root="data/Reddit2", transform=NormalizeFeatures())           
            G_pyg = dataset[0]
            G_nx = self.pyg_to_nx(G_pyg)
        
        elif self.dataset_name == "BlogCatalog":
            
            transform = T.Compose([
                T.NormalizeFeatures(),
                T.LargestConnectedComponents(),
                T.RandomNodeSplit(num_val=500, num_test=500)])
            dataset = AttributedGraphDataset(
                root="data/BlogCatalog",
                name="BlogCatalog",
                transform=transform   
            )
            G_pyg = dataset[0]   
            G_nx = self.pyg_to_nx(G_pyg)
        elif self.dataset_name == "Facebook":
            transform = T.Compose([
                T.NormalizeFeatures(),
                T.LargestConnectedComponents(),
                T.RandomNodeSplit(num_val=500, num_test=500) 
            ])
            dataset = AttributedGraphDataset(
                root="data/Facebook",
                name="Facebook",
                transform=transform
            )
            G_pyg = dataset[0]   
            G_nx = self.pyg_to_nx(G_pyg)

        elif self.dataset_name == "PPI":
            dataset = AttributedGraphDataset(
                root="data/PPI",
                name="PPI",
                transform=T.Compose([
                    # T.NormalizeFeatures(),
                    T.LargestConnectedComponents()
                ])
            )
            G_pyg = dataset[0]
            if G_pyg.y.dim() > 1 and G_pyg.y.shape[1] > 1:
                print(f"Original PPI labels shape: {G_pyg.y.shape}")
                class_counts = G_pyg.y.sum(dim=0)
                most_freq_class_idx = class_counts.argmax().item()
                print(f"PPI Conversion: Selected Class {most_freq_class_idx} (Count: {int(class_counts[most_freq_class_idx])}) as the binary target.")
                
                G_pyg.y = G_pyg.y[:, most_freq_class_idx].long()
            
            
            if not hasattr(G_pyg, 'train_mask') or G_pyg.train_mask is None:
                G_pyg = T.RandomNodeSplit(
                    split='train_rest', 
                    num_val=0.1, 
                    num_test=0.1
                )(G_pyg)

            G_nx = to_networkx(G_pyg, to_undirected=True)

        elif self.dataset_name == "Flickr":

            dataset = AttributedGraphDataset(root="data/Flickr_AGD", name="Flickr")
            G_pyg = dataset[0]
            G_nx = self.pyg_to_nx(G_pyg)
            
        elif self.dataset_name == "KarateClub":
            G_nx = nx.karate_club_graph()
            
            labels = nx.get_node_attributes(G_nx, 'club')

            nodes_by_label = {}
            for node, label in labels.items():
                shifted_node = node 
                if label not in nodes_by_label:
                    nodes_by_label[label] = []
                nodes_by_label[label].append(shifted_node)

        
            instructor_node = nodes_by_label.get('Mr. Hi', [])
            if instructor_node:
                print(f"The instructor node (Mr. Hi) is: {instructor_node[0]}")
            else:
                print("Instructor node not found.")

            officer_node = nodes_by_label.get('Officer', [])
            if officer_node:
                print(f"The officer node (Officer) is: {officer_node[0]}")
            else:
                print("Officer node not found.")
        
            G_pyg = self.nx_to_pyg(G_nx)
            dataset = KarateClub()
            G_pyg = dataset[0]
           
        self.print_graph_statistics(G_nx)
        self.print_pyg_graph_statistics(G_pyg)

        return G_nx, G_pyg

    def compute_homophily(self, G_pyg):
        if not hasattr(G_pyg, 'y') or G_pyg.y is None:
            return None
        edge_index = G_pyg.edge_index
        y = G_pyg.y.cpu().numpy()
        same_label = y[edge_index[0]] == y[edge_index[1]]
        return float(np.sum(same_label)) / edge_index.size(1)
    
    def print_pyg_graph_statistics(self, data):
        """
        Prints statistics of the PyTorch Geometric graph.

        Args:
            data: PyTorch Geometric Data object.
        """
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        print(f'Number of graphs: {len(data)}')
        num_features = data.x.shape[1] if data.x is not None else 0
        print(f"PyG - Number of features per node: {num_features}")
        print(f'Number of classes: {torch.unique(data.y).numel()}') 

        print(f"PyG - Number of nodes: {num_nodes}")
        print(f"PyG - Number of edges: {num_edges}")
        
        print(f'Is undirected: {data.is_undirected()}')
        
        num_features = data.x.shape[1] if data.x is not None else 0
        print(f"PyG - Number of features per node: {num_features}")

        '''homophily = self.compute_homophily(data)
        if homophily is not None:
            print(f"PyG - Homophily: {homophily:.4f}")
        else:
            print("PyG - Homophily: not available (no labels).")


        if hasattr(data, "x") and data.x is not None:
            num_features_total = data.x.numel()
            num_nonzero = (data.x != 0).sum().item()
            sparsity = 1 - num_nonzero / num_features_total
            print(f"PyG - Feature sparsity: {sparsity:.4f}")
        else:
            print("PyG - Feature sparsity: not available (no x).")'''

    
    def compute_avg_ppr_std(self, G, alpha=0.85):
        """
        Compute the exact Average Personalized PageRank (PPR) Standard Deviation
        for all nodes in the graph (no sampling).

        Args:
            G (nx.Graph): The input graph.
            alpha (float): Damping factor (default 0.85).

        Returns:
            float: The average PPR standard deviation.
        """
        ppr_stds = []
        for node in G.nodes():
            try:
                ppr = nx.pagerank(G, alpha=alpha, personalization={node: 1})
                ppr_std = np.std(list(ppr.values()))
                ppr_stds.append(ppr_std)
            except Exception as e:
                print(f"Skipping node {node}: {e}")
                continue

        avg_ppr_std = np.mean(ppr_stds) if ppr_stds else 0.0
        return avg_ppr_std

    def print_graph_statistics(self, G_nx):
        """
        Prints statistics of the NetworkX graph.

        Args:
            G_nx: NetworkX graph object.
        """
        num_nodes = G_nx.number_of_nodes()
        num_edges = G_nx.number_of_edges()
        print(f"NetworkX - Number of nodes: {num_nodes}")
        print(f"NetworkX - Number of edges: {num_edges}")

        is_directed = G_nx.is_directed()
        print(f"NetworkX graph is {'directed' if is_directed else 'undirected'}.")
        is_connected = nx.is_connected(G_nx)
        print(f"NetworkX graph is connected: {is_connected}")

        '''graph_density = nx.density(G_nx)
        print(f"NetworkX graph density: {graph_density:.4f}")

        if num_nodes > 0: 
            node_type = type(next(iter(G_nx.nodes())))  
            print(f"NetworkX - Node ID Type: {node_type}")
        else:
            print("No nodes in the NetworkX graph.")


        if nx.is_connected(G_nx):
            diameter = nx.diameter(G_nx)
        else:
            largest_cc = max(nx.connected_components(G_nx), key=len)
            subgraph = G_nx.subgraph(largest_cc)
            diameter = nx.diameter(subgraph)
        print(f"NetworkX - Diameter (largest connected component): {diameter}")

        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
        print(f"NetworkX - Avg. Degree: {avg_degree:.4f}")

        avg_neighbor_degree = np.mean(list(nx.average_neighbor_degree(G_nx).values()))
        print(f"NetworkX - Avg. Neighbor Degree: {avg_neighbor_degree:.4f}")

        avg_clustering = nx.average_clustering(G_nx)
        print(f"NetworkX - Avg. Clustering Coef.: {avg_clustering:.4f}")

        neighbor_clustering = nx.average_clustering(G_nx)
        neighbor_clustering_dict = nx.clustering(G_nx)
        avg_neighbor_clustering = np.mean([
            np.mean([neighbor_clustering_dict[nbr] for nbr in G_nx.neighbors(node)])
            if len(list(G_nx.neighbors(node))) > 0 else 0
            for node in G_nx.nodes()
        ])
        print(f"NetworkX - Avg. Neighbor Clustering Coef.: {avg_neighbor_clustering:.4f}")

        num_triangles = sum(nx.triangles(G_nx).values()) // 3
        print(f"NetworkX - Number of Triangles: {num_triangles}")

        triangle_dict = nx.triangles(G_nx)
        avg_triangles = np.mean(list(triangle_dict.values()))
        print(f"NetworkX - Avg. Triangles per Node: {avg_triangles:.4f}")


        compute_avg_ppr_std = self.compute_avg_ppr_std(G_nx, alpha=0.85)

        print(f"NetworkX - Avg. PPR std: {compute_avg_ppr_std:.4f}")'''












