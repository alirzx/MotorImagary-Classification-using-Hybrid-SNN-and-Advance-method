import torch
import torch.nn as nn
import numpy as np
import os

# ----------------------------
# LIF neuron
# ----------------------------
class LIFNeuronCell(nn.Module):
    def __init__(self, threshold=0.3, decay=0.9, temp=1.2):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.temp = temp

    def forward(self, x):
        mem_pot = x * self.decay
        spike = (mem_pot > self.threshold).float()
        surrogate = torch.sigmoid((mem_pot - self.threshold) * self.temp)
        spike = spike + surrogate - surrogate.detach()
        return spike


# ----------------------------
# Spiking Multi-Head Attention
# ----------------------------
class SpikingMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, threshold=0.3, decay=0.9, temp=1.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.spike = LIFNeuronCell(threshold=threshold, decay=decay, temp=temp)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.spike(attn_output)


# ----------------------------
# Positional encoding
# ----------------------------
def positional_encoding(seq_len, d_model, device):
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(d_model // 2, dtype=torch.float32, device=device)
    angle_rates = 1 / (10000 ** (2 * i / d_model))
    angle_rads = pos * angle_rates
    sin = torch.sin(angle_rads)
    cos = torch.cos(angle_rads)
    pos_encoding = torch.cat([sin, cos], dim=-1)
    return pos_encoding.unsqueeze(0)


# ----------------------------
# SpiTranNet compatible with checkpoint
# ----------------------------
class SpiTranNet(nn.Module):
    def __init__(
        self,
        input_channels=22,
        input_length=1000,
        num_classes=2,
        num_heads=2,
        threshold=0.3,
        decay=0.9,
        temp=1.2,
        dropout=0.5
    ):
        super().__init__()
        self.input_length = input_length
        self.transformer_dim = 128
        self.seq_len = input_length // 64

        # CNN feature extractor
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout)
        )

        # Transformer encoder layers matching checkpoint names
        self.attn = SpikingMultiHeadAttention(self.transformer_dim, num_heads, threshold, decay, temp)
        self.norm1 = nn.LayerNorm(self.transformer_dim)

        # ffn exactly matching checkpoint
        self.ffn = nn.Sequential(
            nn.Linear(self.transformer_dim, 64),  # checkpoint has 128->64
            nn.ReLU(),
            nn.Linear(64, self.transformer_dim)   # checkpoint has 64->128
        )
        # if LIFNeuronCell was outside sequential in training, insert here:
        self.lif = LIFNeuronCell(threshold, decay, temp)

        self.norm2 = nn.LayerNorm(self.transformer_dim)

        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.seq_len * self.transformer_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)  # [B, Seq, Embed]
        pos_enc = positional_encoding(x.size(1), x.size(2), x.device)
        x = x + pos_enc

        # Transformer
        res = x
        x = self.attn(x)
        x = self.norm1(x + res)

        res = x
        x = self.ffn(x)
        x = self.lif(x)   # if used outside sequential
        x = self.norm2(x + res)

        x = self.dropout(x)
        x = self.flatten(x)
        return self.classifier(x)

# ----------------------------
# API helper functions
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_input(x):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x.to(device)


def load_model(model_path, input_channels=22, input_length=1000, num_classes=2):
    model = SpiTranNet(input_channels, input_length, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_all_subject_models(subject_model_paths):
    models = {}
    for subj, path in subject_model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        models[subj] = load_model(path)
    return models
