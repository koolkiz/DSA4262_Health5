from attn import CrossAttentionFusion
import torch.nn as nn

class MultimodalClassifier(nn.Module):
    def __init__(self, img_dim, csv_dim, embed_dim, num_classes):
        super(MultimodalClassifier, self).__init__()
        self.fusion_layer = CrossAttentionFusion(img_dim, csv_dim, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, img_features, csv_features):
        fused_rep = self.fusion_layer(img_features, csv_features)
        logits = self.classifier(fused_rep)
        return logits
