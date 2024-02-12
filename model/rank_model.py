from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import numpy as np

from model.reference_encoder import Conv_Net


class AdditiveAttention(nn.Module):
    def __init__(self, dropout, query_vector_dim, candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, candidate_vector):
        temp = torch.tanh(self.linear(candidate_vector))
        candidate_weights = F.softmax(
            torch.matmul(temp, self.attention_query_vector), dim=1
        )
        candidate_weights = self.dropout(candidate_weights)

        target = torch.bmm(
            candidate_weights.unsqueeze(dim=1), candidate_vector
        ).squeeze(dim=1)
        return target


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x


class IntensityExtractor(nn.Module):
    def __init__(
        self,
        mel_dim=80,
        pitch_dim=128,
        energy_dim=128,
        fft_dim=128,
        num_heads=8,
        # num_layers=4,
        kernel_size=1,
        n_emotion=5,
        n_bins=10,
        stats_path="datasets/esd_processed/stats.json"
    ):
        super(IntensityExtractor, self).__init__()

        with open(stats_path, "r") as f:
            stats = json.load(f)
            pitch_mean = stats["pitch"]["mean"]
            pitch_std = stats["pitch"]["std"]
            pitch_min = stats["pitch"]["min"]
            pitch_max = stats["pitch"]["max"]
            
            energy_mean = stats["energy"]["mean"]
            energy_std = stats["energy"]["std"]
            energy_min = stats["energy"]["min"]
            energy_max = stats["energy"]["max"]
            
        self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        
        self.pitch_embedding = nn.Embedding(n_bins, pitch_dim)
        self.energy_embedding = nn.Embedding(n_bins, energy_dim)
        
        self.input_projection = nn.Linear(mel_dim + pitch_dim + energy_dim, fft_dim)

        self.pos_enc = PositionalEncoder(fft_dim, 1000)

        self.trans_enc = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=fft_dim,
                nhead=num_heads,
                dim_feedforward=fft_dim * 4,
                batch_first=True,
            ),
            nn.TransformerEncoderLayer(
                d_model=fft_dim,
                nhead=num_heads,
                dim_feedforward=fft_dim * 4,
                batch_first=True,
            ),
            # nn.TransformerEncoderLayer(
            #     d_model=fft_dim,
            #     nhead=num_heads,
            #     dim_feedforward=fft_dim * 4,
            #     batch_first=True,
            # ),
            # nn.TransformerEncoderLayer(
            #     d_model=fft_dim,
            #     nhead=num_heads,
            #     dim_feedforward=fft_dim * 4,
            #     batch_first=True,
            # ),
        )
        self.conv1d = nn.Conv1d(fft_dim, fft_dim, kernel_size, padding=kernel_size // 2)

        self.emotion_embedding = nn.Embedding(n_emotion - 1, fft_dim // 2)

        self.emo_prediction = nn.Linear(fft_dim, n_emotion)

    def forward(self, mel, pitch=None, energy=None, emo_id=None):
        if pitch is None or energy is None:
            x = mel  # (batch, length, channels)
        else:
            p_emb = self.pitch_embedding(
                torch.bucketize(pitch, self.pitch_bins)
            )
            e_emb = self.energy_embedding(
                torch.bucketize(energy, self.energy_bins)
            )
            x = torch.cat([mel, p_emb, e_emb], dim=2)  # (batch, length, channels)

        x = self.input_projection(x)  # (batch, length, fft_dim)

        x = self.pos_enc(x)  # (batch, length, fft_dim)
        x = self.trans_enc(x)  # (batch, length, fft_dim)
        x = x.transpose(1, 2)  # Conv1D expects (batch, fft_dim, length)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # Switch back to (batch, length, fft_dim)

        if emo_id is not None:
            emotion_embed = (
                self.emotion_embedding(emo_id - 1)
                .unsqueeze(1)
                .expand(-1, x.size(1), -1)
            )
            emotion_embed = torch.cat([emotion_embed, emotion_embed], dim=2)

            x = x + emotion_embed

        i = self.emo_prediction(x)

        return i, x   # (batch, length, n_emotion), (batch, length, fft_dim)


class RankModel(nn.Module):
    def __init__(
        self, 
        fft_dim=256, 
        n_emotion=5,
        stats_path="datasets/esd_processed/stats.json",
    ):
        super(RankModel, self).__init__()
        self.intensity_extractor = IntensityExtractor(
            fft_dim=fft_dim,
            n_emotion=n_emotion,
            stats_path=stats_path,
        )
        self.emotion_predictor = nn.Linear(fft_dim, n_emotion)
        self.rank_predictor = nn.Linear(fft_dim, 1)

    def forward(self, x, pitch=None, energy=None, emo_id=None):
        i, x = self.intensity_extractor(x, pitch=pitch, energy=energy, emo_id=emo_id)  # (batch, length, n_emotion)
        h = i.mean(dim=1)  # (batch, n_emotion)
        
        r = self.rank_predictor(x)  # (batch, length)
        r = r.mean(dim=1)

        return (
            i,  # Intensity representations
            h,  # Mean intensity representations
            r,  # Intensity predictions
        )


if __name__ == "__main__":
    model = RankModel()
    x = torch.randn(2, 100, 80)
    emo_class = torch.tensor([1, 2])
    i, h, r = model(x, emo_class)
    print(i.size(), h.size(), r.size())
