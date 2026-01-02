import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class SignEnhancedMessagePassing(MessagePassing):
    """
    Corresponds to Paper Section 4.1.1: Sign-Enhanced Message Passing
    Eq (3) [cite: 163]
    """
    def __init__(self, in_dim, out_dim):
        super(SignEnhancedMessagePassing, self).__init__(aggr='add')
        self.W_u = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        # Simple sign embedding: 0 for negative, 1 for positive
        self.sign_embedding = nn.Embedding(2, 1) 

    def forward(self, x, edge_index, edge_attr):
        # x: node features
        # edge_attr: 0 for negative link, 1 for positive link
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # In paper: Asign * E * W (Eq 3)
        # Here simplified: modulated by learnable sign weight
        sign_weight = self.sign_embedding(edge_attr).view(-1, 1) # (E, 1)
        # Negative edge weights are effectively learned via this embedding
        return sign_weight * self.W_v(x_j)

    def update(self, aggr_out, x):
        # E(u, l) = E(u, l-1) + ...
        return x + F.relu(aggr_out)

class IntentAwareAssignment(nn.Module):
    """
    Corresponds to Paper Section 4.1.2: Factor-Aware Representation Learning
    Eq (1) & (2) [cite: 172, 175]
    """
    def __init__(self, dim, K):
        super(IntentAwareAssignment, self).__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, K)
        )
        self.sign_map = nn.Embedding(2, K) # psi(s_ij)

    def forward(self, x, intent_prototypes):
        # x: (N, dim)
        # intent_prototypes: (K, dim)
        
        # Compute logits for node belonging to each Intent
        # Note: Eq(1) technically depends on edge signs. 
        # Here simplified: Compute Intent distribution based on node features for global view.
        logits = self.mlp(x) # (N, K)
        
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (N, K)
        
        # Compute Intent-Aware Representation Eq(2)
        # r_i = sum(P(c_k|e_i) * c_k)
        # (N, K) @ (K, dim) -> (N, dim)
        r_i = torch.matmul(probs, intent_prototypes)
        
        return r_i, probs

class SignAwareAttentionLayer(nn.Module):
    """
    Corresponds to Paper Section 4.2.2: Sign-Aware Multi-Head Attention
    Eq (7) - (11) [cite: 224, 229, 232, 236, 239]
    """
    def __init__(self, in_dim, out_dim, K_heads=4):
        super(SignAwareAttentionLayer, self).__init__()
        self.K_heads = K_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // K_heads
        
        self.W_l = nn.Linear(in_dim, out_dim)
        self.att_vec = nn.Parameter(torch.Tensor(1, K_heads, self.head_dim))
        self.sign_embed = nn.Embedding(2, self.head_dim) # psi(s_ij)
        
        self.W_head = nn.Linear(out_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.eps = nn.Parameter(torch.Tensor([0]))

        nn.init.xavier_uniform_(self.att_vec)

    def forward(self, x, edge_index, edge_sign):
        # x: (N_sub, in_dim)
        # edge_sign: (E_sub, ) 0 or 1
        
        row, col = edge_index
        
        # Linear transformation
        h = self.W_l(x) # (N, out_dim)
        h_reshaped = h.view(-1, self.K_heads, self.head_dim)
        
        # Source and target node features
        h_i = h_reshaped[row] # (E, K, D_h)
        h_j = h_reshaped[col] # (E, K, D_h)
        
        # Sign Embedding
        s_emb = self.sign_embed(edge_sign).view(-1, 1, self.head_dim) # (E, 1, D_h)
        
        # Attention Calculation Eq (7)
        # e_ij = (alpha)^T * (W h_i || W h_j + psi(s) ...)
        # Simplified here as element-wise combination followed by dot product
        score = (h_i * h_j * s_emb).sum(dim=-1) # (E, K)
        
        # Softmax Eq (8)
        alpha = softmax(score, row, num_nodes=x.size(0)) # (E, K)
        
        # Aggregation Eq (9)
        # Manually implement weighted sum
        out_heads = []
        for k in range(self.K_heads):
            # Aggregate for each head
            aggr = torch.zeros(x.size(0), self.head_dim).to(x.device)
            # scatter_add
            weighted_msg = h_j[:, k, :] * alpha[:, k].view(-1, 1)
            aggr.index_add_(0, row, weighted_msg)
            out_heads.append(aggr)
            
        # Concat heads
        m_l = torch.cat(out_heads, dim=1) # (N, out_dim)
        
        # Linear Fusion Eq (10)
        m_l = self.W_head(m_l)
        
        # GIN style update Eq (11)
        h_next = self.mlp((1 + self.eps) * h + m_l)
        
        return h_next