import torch
import torch.nn as nn
import torch.nn.functional as F

class DiSignLoss(nn.Module):
    def __init__(self, temp_gl=0.2, temp_intent=0.2, margin_sign=0.5):
        super(DiSignLoss, self).__init__()
        self.temp_gl = temp_gl
        self.temp_intent = temp_intent
        self.margin_sign = margin_sign
        self.bce = nn.BCELoss()

    def global_local_contrastive(self, global_rep, local_rep):
        """
        Eq (13): Global-Local Contrastive Loss [cite: 261]
        """
        # Normalize
        global_rep = F.normalize(global_rep, dim=1)
        local_rep = F.normalize(local_rep, dim=1)
        
        # Positive pairs: same node, global vs local
        pos_score = torch.sum(global_rep * local_rep, dim=1) / self.temp_gl
        pos_score = torch.exp(pos_score)
        
        # All pairs denominator (In-batch negatives)
        # Matmul (B, D) @ (D, B) -> (B, B)
        all_scores = torch.matmul(global_rep, local_rep.t()) / self.temp_gl
        denom = torch.sum(torch.exp(all_scores), dim=1)
        
        loss = -torch.log(pos_score / denom).mean()
        return loss

    def intent_clustering_loss(self, probs):
        """
        Eq (16): Intent-Level Clustering Loss [cite: 271]
        KL Divergence between Q (probs) and P (target distribution)
        """
        # P computation Eq (15)
        q = probs
        p_numer = q**2 / q.sum(0)
        p_denom = p_numer.sum(1, keepdim=True)
        p = p_numer / p_denom
        
        # KL Divergence
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss

    def sign_triplet_loss(self, z_u, z_v_pos, z_v_neg):
        """
        Eq (17): Sign-Aware Triplet Contrast [cite: 278]
        Maximize similarity with positive neighbors, minimize with negative
        """
        pos_sim = F.cosine_similarity(z_u, z_v_pos)
        neg_sim = F.cosine_similarity(z_u, z_v_neg)
        
        # Margin loss: max(0, margin - (pos - neg)) -> max(0, neg - pos + margin)
        # Note: Paper says max(0, eta - sim_pos + sim_neg) which is standard triplet
        loss = torch.clamp(self.margin_sign - pos_sim + neg_sim, min=0).mean()
        return loss

    def forward(self, pred, target, gl_pairs, intent_probs, triplet_pairs):
        """
        Combined Loss Eq (23) [cite: 319]
        """
        # 1. Prediction Loss (Weighted BCE)
        # target: 1 for positive, 0 for negative
        loss_pred = self.bce(pred, target.float())
        
        # 2. Global-Local
        loss_gl = self.global_local_contrastive(gl_pairs[0], gl_pairs[1])
        
        # 3. Intent Clustering
        loss_intent = self.intent_clustering_loss(intent_probs)
        
        # 4. Sign Triplet
        loss_sign = self.sign_triplet_loss(triplet_pairs[0], triplet_pairs[1], triplet_pairs[2])
        
        return loss_pred, loss_gl, loss_intent, loss_sign
