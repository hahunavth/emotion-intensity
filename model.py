from typing import Dict, List, Tuple, Union
from sympy import fft
import torch
import torch.nn as nn
import torch.nn.functional as F

from reference_encoder import Conv_Net


class AdditiveAttention(nn.Module):
    def __init__(self, dropout,
                 query_vector_dim,
                 candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, candidate_vector):
        temp = torch.tanh(self.linear(candidate_vector))
        candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector),dim=1)
        candidate_weights = self.dropout(candidate_weights)
            
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target



class IntensityExtractor(nn.Module):
    def __init__(
        self, 
        mel_dim=80, 
        pitch_dim=0, 
        energy_dim=0, 
        fft_dim=256, 
        num_heads=8, 
        # num_layers=4, 
        kernel_size=1,
        n_emotion=5,
    ):
        super(IntensityExtractor, self).__init__()
        emotion_embedding_dim = fft_dim // 2
        self.emotion_embedding = nn.Embedding(n_emotion-1, emotion_embedding_dim)
        # self.emotion_embedding = nn.Parameter(torch.randn(5, emotion_embedding_dim))
        self.input_projection = nn.Linear(mel_dim + pitch_dim + energy_dim, fft_dim)

        self.trans_enc = nn.TransformerEncoderLayer(d_model=fft_dim, nhead=num_heads, dim_feedforward=fft_dim*4, batch_first=True)
        self.conv1d = nn.Conv1d(fft_dim, fft_dim, kernel_size, padding=kernel_size // 2)
        
        # self.emotion_prediction = nn.Linear(fft_dim, 5)
        # self.feature_projection = nn.Linear(fft_dim, fft_dim)

    def forward(self, mel, pitch=None, energy=None, emo_class=None):
        if pitch is None or energy is None:
            x = mel            # (batch, length, channels)
        else:
            x = torch.cat([mel, pitch, energy], dim=-1)
        x = self.input_projection(x) # (batch, length, fft_dim)
        
        x = self.trans_enc(x)  # (batch, length, fft_dim)
        x = x.transpose(1, 2)  # Conv1D expects (batch, fft_dim, length)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # Switch back to (batch, length, fft_dim)
        
        if emo_class is not None:
            emotion_embed = self.emotion_embedding(emo_class-1).unsqueeze(1).expand(-1, x.size(1), -1)
            emotion_embed = torch.cat([emotion_embed, emotion_embed], dim=2)
        
            x = x + emotion_embed
            # x = torch.cat([x, emotion_embed], dim=-1)
        
        return x               # (batch, length, fft_dim)


class RankModel(nn.Module):
    def __init__(self, fft_dim=256, n_emotion=5):
        super(RankModel, self).__init__()
        self.intensity_extractor = IntensityExtractor(
            fft_dim=fft_dim,
            n_emotion=n_emotion,
        )
        self.rank_predictor = nn.Sequential(
            nn.Linear(fft_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.emotion_predictor = nn.Linear(fft_dim, n_emotion)

    def forward(self, x, emo_class=None):
        i = self.intensity_extractor(x, emo_class)  # (batch, length, fft_dim)
        
        _h = i.mean(dim=1)                          # (batch, fft_dim)
        h = self.emotion_predictor(_h)              # (batch, n_emotion)
        
        r = self.rank_predictor(_h)                 # (batch, 1)
        
        return (
            i,  # Intensity representations
            h,  # Mean intensity representations
            r,  # Intensity predictions
            _h,
        )


__factory_model = {
    'rank': RankModel,
}

__args_dict_model = {
    'rank': {},
}


def create_model(name: str, **kwargs):
    assert name in __factory_model, f'invalid model_name: {name}'
    _kwargs = {k: v for k, v in kwargs.items() if k in __args_dict_model[name]}
    default_kwargs = __args_dict_model[name]
    new_kwargs = {**default_kwargs, **_kwargs}
    model = __factory_model[name](**new_kwargs)
    
    return model, new_kwargs


if __name__ == "__main__":
    pass
    # intensity_extractor = IntensityExtractor()
    # mel = torch.rand(3, 120, 80) 
    # print(intensity_extractor(mel, emo_class=torch.LongTensor([1, 1, 2])).shape)
    
    # print(create_model('rank'))

    # # Initialize parameters
    # mel_dim = 80  # Example dimension for Mel-Spectrogram
    # pitch_dim = 1  # Pitch is typically a single value per time step
    # energy_dim = 1  # Energy is typically a single value per time step
    # emotion_embedding_dim = 512  # Example dimension for emotion embeddings
    # fft_dim = 512  # Dimensionality of FFT blocks
    # num_heads = 8  # Number of heads in multi-head attention mechanism
    # num_layers = 4  # Number of FFT layers
    # emotion_classes = 5  # Number of different emotions
    # kernel_size = 3  # Kernel size for 1D convolution

    # # Create an instance of the IntensityExtractor
    # model = IntensityExtractor(mel_dim, pitch_dim, energy_dim, emotion_embedding_dim, fft_dim, num_heads, num_layers, kernel_size)

    # # Example inputs (batch_size = 32, sequence_length = 100)
    # mel = torch.rand(32, 100, mel_dim)  # Mel-Spectrogram
    # pitch = torch.rand(32, 100, pitch_dim)  # Pitch contour
    # energy = torch.rand(32, 100, energy_dim)  # Energy
    # emotion_class = torch.randint(low=0, high=emotion_classes, size=(32,))  # Emotion class labels

    # # Forward pass
    # intensity_representation, emotion_prediction = model(mel, pitch, energy)

    # # Output
    # print(intensity_representation.shape)  # Expected shape: (batch_size, fft_dim)
    # print(emotion_prediction.shape)