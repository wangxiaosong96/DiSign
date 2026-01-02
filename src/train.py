import torch
from torch.utils.data import DataLoader
from src.dataset import SignedBipartiteDataset, collate_fn
from src.model import DiSign
from src.loss import DiSignLoss
import torch.optim as optim

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001
LAMBDAS = [1.0, 0.1, 0.1, 0.1] # weights for pred, gl, intent, sign

# 1. Mock Data (Need real data here)
# (u, v, sign)
raw_edges = [(0, 1, 1), (0, 2, -1), (1, 3, 1), (2, 3, -1), (0, 3, 1)] * 20
num_nodes = 50 # Total nodes in bipartite graph

dataset = SignedBipartiteDataset(raw_edges, num_nodes)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 2. Initialize Model
model = DiSign(
    num_users=num_nodes, 
    num_items=num_nodes, 
    embed_dim=64, 
    K_intents=4
).to(DEVICE)

# Pre-compute initial global embeddings (Simulated)
# In real code, run the Global GNN on the whole graph once per epoch or use mini-batch neighbor sampling
global_u_emb = torch.randn(num_nodes, 64).to(DEVICE)
global_v_emb = torch.randn(num_nodes, 64).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = DiSignLoss()

# 3. Training Loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        
        # Get root nodes for global lookup
        # dataset.py needs to pass root_u/v indices properly
        # assuming batch.root_u maps back to global index
        u_idx = torch.tensor(batch.root_u).long().to(DEVICE)
        v_idx = torch.tensor(batch.root_v).long().to(DEVICE)
        
        # Forward
        pred, r_u, h_sub, probs = model(u_idx, v_idx, global_u_emb, global_v_emb, batch)
        
        # Construct Contrastive Pairs (Simplified for demo)
        # GL Pairs: Global r_u vs Local h_sub (Need dimension alignment in real impl)
        gl_pairs = (r_u, h_sub) 
        
        # Triplet Pairs: (Anchor, Pos, Neg) - requires sampling triplets
        # Using dummy for code to run; in practice you need to sample pos/neg neighbors for u
        triplet_pairs = (r_u, r_u, r_u) 
        
        # Calculate Loss
        l_pred, l_gl, l_intent, l_sign = loss_fn(
            pred, batch.y, gl_pairs, probs, triplet_pairs
        )
        
        loss = LAMBDAS[0]*l_pred + LAMBDAS[1]*l_gl + LAMBDAS[2]*l_intent + LAMBDAS[3]*l_sign
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

# Save
torch.save(model.state_dict(), "disign_model.pth")