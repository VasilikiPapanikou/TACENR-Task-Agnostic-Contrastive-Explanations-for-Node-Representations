# TACENR: Task-Agnostic Contrastive Explanations for Node Representations

Graph representation learning has achieved notable success in encoding graph-structured data into latent vector spaces, enabling a wide range of downstream tasks. However, these node representations remain opaque and difficult to interpret. Existing explainability methods primarily focus on supervised settings or on explaining individual representation dimensions, leaving a critical gap in explaining the overall structure of node representations. In this paper, we propose TACENR (Task-Agnostic Contrastive Explanations for Node Representations), a local explanation method that identifies not only attribute features but also proximity and structural ones that contribute the most in the representation space. 
TACENR builds on contrastive learning, through which we learn a similarity function in the representation space, revealing which are the features that play an important role in the representation of a node. 
While our focus is on task-agnostic explanations, TACENR can be applied to supervised scenarios as well. Experimental results demonstrate that proximity and structural features play a significant role in shaping node representations and that our supervised variant performs comparably to existing task-specific approaches in identifying the most impactful features. 

<img width="3127" height="845" alt="pipeline_TACENR" src="https://github.com/user-attachments/assets/6e0b59fe-d110-4eca-8faa-57d4c7d157d8" />


### Repository Structure:

- src/: Contains all Python scripts for loading datasets and generating explanations using the TACENR explainer.
- data/: Contains the datasets
- supervised_embeddings/: Precomputed supervised embeddings stored for all datasets and models.
- unsupervised_embeddings/: Precomputed supervised embeddings stored for all datasets and models.
- aopc_metric.ipynb: Notebook for calculating AOPC metric.
- aopc_visualization_all_models_datasets.ipynb: Notebook for generating AOPC plots across all models and datasets.
