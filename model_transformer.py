import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerServiceClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_heads=4, num_layers=2, output_dim=15, dropout=0.3):
        super().__init__()
        
        # Project embedding to hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding (optional, but helps)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch_size, embedding_dim)
        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Project to hidden dimension
        x = self.input_projection(x)  # (batch_size, 1, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)  # (batch_size, 1, hidden_dim)
        
        # Take the [CLS] token representation
        x = x[:, 0, :]  # (batch_size, hidden_dim)
        
        # Classification
        output = self.classifier(x)  # (batch_size, output_dim)
        
        return output

# Test model creation
if __name__ == "__main__":
    model = TransformerServiceClassifier(
        input_dim=384,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        output_dim=15,
        dropout=0.3
    )
    batch = torch.randn(8, 384)
    output = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Model created successfully")
    