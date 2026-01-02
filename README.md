# DiSign
## This repository contains the PyTorch implementation of the paper: "DiSign: Disentangled Sign-Aware Representation Learning for Robust Recommendation in Signed Bipartite Graphs".


![image](https://github.com/wangxiaosong96/DiSign/blob/main/graph/graph3.png)

##📝 AbstractNode representation learning in signed bipartite graphs is critical for applications such as recommender systems (predicting likes/dislikes).
Existing methods often rely on simplistic sociological theories (e.g., Balance Theory), which fail to capture the complex nature of real-world interactions.
DiSign is a novel framework that transcends these constraints by attributing user-item interactions to independent latent intents. 
It incorporates three synergistic components:Global Disentangled Intent Encoder: Decomposes node embeddings into $K$ intent-specific representations.
Local Sign-Aware Subgraph Extractor: Captures nuanced topological patterns using a novel Two-Anchor Distance Labeling encoding.
Multi-Granularity Contrastive Learning: Applies self-supervision at the node, intent, and subgraph levels to enhance robustness against data sparsity.

##🚀 Key FeaturesTheory-Agnostic: 
Does not rely on rigid Balance Theory assumptions, making it more flexible for complex real-world data.
Disentangled Representation: Explicitly models latent factors (e.g., price, quality, brand) driving user-item interactions.
Sign-Awareness: Distinguishes between positive and negative edges during both message passing and attention aggregation.
Robustness: State-of-the-art performance on Review, Bonanza, ML-1M, and Amazon-Book datasets, especially under sparse data conditions.



## Requirements
python 3.7.1
torch>=1.12.0
torch_geometric>=2.3.0
numpy
scipy
networkx
scikit-learn
tqdm


##🤝 Citation
If you find this code useful for your research, please cite our paper:
@article{wang2025disign,
  title={DiSign: Disentangled Sign-Aware Representation Learning for Robust Recommendation in Signed Bipartite Graphs},
  author={Wang, Xiaosong and Wang, Qingyong and Gu, Lichuan},
  journal={Manuscript},
  year={2025},
  institution={Anhui Agricultural University}
}

## Contact

Please feel free to contact us if you need any help: xiaosongwang@ahau.edu.cn
