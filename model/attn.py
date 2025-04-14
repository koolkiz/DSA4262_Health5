import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, img_dim, csv_dim, embed_dim, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        
        # Linear projections to match embedding dimensions
        self.img_proj = nn.Linear(img_dim, embed_dim)
        self.csv_proj = nn.Linear(csv_dim, embed_dim)
        
        # Cross-attention layers
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Feed-forward layers
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, img_features, csv_features):
        """
        img_features: Tensor of shape (batch_size, img_dim)
        csv_features: Tensor of shape (batch_size, csv_dim)
        """
        
        # Project features into the embedding space
        img_emb = self.img_proj(img_features).unsqueeze(0)  # Shape: (1, batch, embed_dim)
        csv_emb = self.csv_proj(csv_features).unsqueeze(0)  # Shape: (1, batch, embed_dim)
        
        # Cross-attention: Image attends to CSV features
        attn_out_1, _ = self.cross_attn_1(img_emb, csv_emb, csv_emb)
        attn_out_1 = self.norm1(img_emb + self.dropout(attn_out_1))
        
        # Cross-attention: CSV features attend to Image
        attn_out_2, _ = self.cross_attn_2(csv_emb, img_emb, img_emb)
        attn_out_2 = self.norm2(csv_emb + self.dropout(attn_out_2))
        
        # Fusion via element-wise sum
        fused_rep = attn_out_1 + attn_out_2  # Shape: (1, batch, embed_dim)
        fused_rep = fused_rep.squeeze(0)  # Shape: (batch, embed_dim)
        
        # Final feed-forward processing
        fused_rep = self.ffn(fused_rep)
        return fused_rep