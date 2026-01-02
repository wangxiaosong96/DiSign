import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import numpy as np

class SignedBipartiteDataset(Dataset):
    def __init__(self, edge_list, num_nodes, k_hop=2):
        """
        edge_list: [(u, v, sign), ...]
        """
        super().__init__()
        self.edge_list = edge_list
        self.k_hop = k_hop
        self.num_nodes = num_nodes
        
        # Build graph for extraction
        self.G = nx.Graph()
        for u, v, s in edge_list:
            self.G.add_edge(u, v, sign=s)
            
        # Node features (Identity for now)
        self.node_features = torch.eye(num_nodes)[:, :64] # Random 64-dim feat

    def len(self):
        return len(self.edge_list)

    def get(self, idx):
        # Target Link
        u, v, label = self.edge_list[idx]
        
        # --- 1. Subgraph Extraction ---
        # Get k-hop neighbors for u and v (Eq 4)
        nodes_u = nx.single_source_shortest_path_length(self.G, u, cutoff=self.k_hop)
        nodes_v = nx.single_source_shortest_path_length(self.G, v, cutoff=self.k_hop)
        
        subgraph_nodes = set(nodes_u.keys()) | set(nodes_v.keys())
        subgraph_nodes = list(subgraph_nodes)
        
        # Mapping to local index
        mapping = {node: i for i, node in enumerate(subgraph_nodes)}
        
        # --- 2. Two-Anchor Distance Labeling (Eq 6) ---
        dist_labels = []
        d_max = 2 * self.k_hop # Approximation
        
        for node in subgraph_nodes:
            d_xu = nodes_u.get(node, 999) # 999 is infinity
            d_xv = nodes_v.get(node, 999)
            
            # Simplified Logic for Eq (6) [cite: 199]
            f_label = 1 + min(d_xu, d_xv) + (d_xu + d_xv) * d_max // 2
            # Clip label to avoid index error
            f_label = min(f_label, 9) 
            dist_labels.append(f_label)
            
        dist_labels = torch.tensor(dist_labels, dtype=torch.long)
        
        # --- 3. Build PyG Data Object ---
        src_row = []
        dst_col = []
        edge_attrs = []
        
        sub_G = self.G.subgraph(subgraph_nodes)
        for i, j in sub_G.edges():
            if i in mapping and j in mapping:
                src = mapping[i]
                dst = mapping[j]
                # Undirected for GNN
                src_row.extend([src, dst])
                dst_col.extend([dst, src])
                
                sign = sub_G[i][j]['sign']
                # Map sign: -1 -> 0, 1 -> 1
                s_val = 1 if sign > 0 else 0
                edge_attrs.extend([s_val, s_val])
                
        edge_index = torch.tensor([src_row, dst_col], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        x = self.node_features[subgraph_nodes]
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.dist_label = dist_labels
        data.y = torch.tensor([1.0 if label > 0 else 0.0], dtype=torch.float)
        data.root_u = u
        data.root_v = v
        
        return data

def collate_fn(batch):
    # Use PyG Batch
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)

