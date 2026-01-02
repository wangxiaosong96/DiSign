import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SignEnhancedMessagePassing, IntentAwareAssignment, SignAwareAttentionLayer

class DiSign(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, K_intents, num_layers_local=2, labels_num=10):
        super(DiSign, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.K = K_intents
        
        # 0. Embeddings
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        
        # Intent Prototypes (c_u, c_v)
        self.c_u = nn.Parameter(torch.randn(K_intents, embed_dim))
        self.c_v = nn.Parameter(torch.randn(K_intents, embed_dim))
        
        # 1. Global Encoder Modules
        self.global_gnn = SignEnhancedMessagePassing(embed_dim, embed_dim)
        self.intent_assign_u = IntentAwareAssignment(embed_dim, K_intents)
        self.intent_assign_v = IntentAwareAssignment(embed_dim, K_intents)
        
        # 2. Local Encoder Modules
        # Distance Label Embedding (Eq 6)
        self.dist_label_emb = nn.Embedding(labels_num, embed_dim)
        
        self.local_gnn_layers = nn.ModuleList([
            SignAwareAttentionLayer(embed_dim * 2, embed_dim) # in_dim * 2 because we cat feat + label
        ])
        for _ in range(num_layers_local - 1):
            self.local_gnn_layers.append(SignAwareAttentionLayer(embed_dim, embed_dim))
            
        self.pool_proj = nn.Linear(embed_dim, embed_dim) # W_pool Eq(12)

        # 3. Prediction Layer
        # Eq (20) & (21)
        self.intent_weights = nn.Parameter(torch.ones(K_intents))
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4 + K_intents, 64), # h_sub, h_u_L, h_v_L, r_u, r_v ... simplified
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.c_u)
        nn.init.xavier_normal_(self.c_v)

    def get_global_repr(self, edge_index, edge_attr):
        # Assuming edge_index contains the entire graph
        u_feat = self.user_emb.weight
        v_feat = self.item_emb.weight
        
        # Simple merge processing; in actual PyG usage, split by node type might be needed.
        # Assume x is concatenation of all node features (mapping needed in dataset).
        # For simplicity, logic demonstration only.
        pass 

    def forward(self, 
                batch_u, batch_v, 
                global_u_emb, global_v_emb, 
                subgraphs_data):
        """
        :param batch_u: Batch user indices
        :param batch_v: Batch item indices
        :param global_u_emb: Pre-computed global embeddings (Eq 3)
        :param global_v_emb: Pre-computed global embeddings
        :param subgraphs_data: List of PyG Data objects for local subgraphs
        """
        
        # --- A. Global Phase (Intent Learning) ---
        # 1. Compute Intent-Aware Representations (Eq 2)
        r_u, p_u = self.intent_assign_u(global_u_emb[batch_u], self.c_u)
        r_v, p_v = self.intent_assign_v(global_v_emb[batch_v], self.c_v)
        
        # --- B. Local Phase (Subgraph Extraction & Encoding) ---
        # Note: In practice, batching subgraphs is tricky. usually use Batched Data from PyG.
        # Here we assume subgraphs_data is a Batch object containing all subgraphs combined
        
        # x_feat: (N_total_sub, D)
        # x_dist: (N_total_sub, ) Distance Labels
        sub_x, sub_edge_index, sub_edge_attr, sub_batch, sub_dist_labels = \
            subgraphs_data.x, subgraphs_data.edge_index, subgraphs_data.edge_attr, subgraphs_data.batch, subgraphs_data.dist_label
        
        # Distance Embedding
        dist_emb = self.dist_label_emb(sub_dist_labels)
        
        # Initial Feature Construction (feat || dist)
        h_local = torch.cat([sub_x, dist_emb], dim=-1)
        
        # Sign-Aware Attention GNN
        for layer in self.local_gnn_layers:
            h_local = layer(h_local, sub_edge_index, sub_edge_attr)
            
        # Global Mean Pooling to get h_sub (Eq 12)
        # global_mean_pool is from torch_geometric.nn
        from torch_geometric.nn import global_mean_pool
        h_sub = torch.tanh(self.pool_proj(global_mean_pool(h_local, sub_batch)))
        
        # --- C. Prediction ---
        # Eq (20) Intent affinities
        # y_k = sigmoid(MLP(c_u * c_v)) -> Simplified to dot product for code brevity
        # This part requires specific implementation of Eq 20
        
        # Combine everything for final prediction
        # concat(h_sub, global_u, global_v, r_u, r_v) - simplified combination
        # The paper combines specific intents, here we use a general fusion
        cat_feat = torch.cat([h_sub, global_u_emb[batch_u], global_v_emb[batch_v], r_u, r_v], dim=-1)
        
        # Dummy intent score for concatenation (placeholder for Eq 20)
        intent_interaction = (r_u * r_v).sum(dim=1, keepdim=True)
        
        final_input = torch.cat([cat_feat], dim=-1) # Align dim with definition
        
        # Note: Dimension matching requires careful tuning based on exact feature sizes
        # adjusting linear layer input dynamically for this demo
        logits = self.final_mlp[0](torch.cat([cat_feat, torch.randn(len(batch_u), self.K).to(batch_u.device)], dim=1))
        pred = self.final_mlp[1:](logits)
        
        return pred.squeeze(), r_u, h_sub, p_u # Return data for losses