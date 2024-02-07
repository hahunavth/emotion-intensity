from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.reference_encoder import Conv_Net


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
    def __init__(self, mel_dim, pitch_dim, energy_dim, emotion_embedding_dim, fft_dim, num_heads, num_layers, kernel_size=1):
        super(IntensityExtractor, self).__init__()
        # self.emotion_embedding = nn.Embedding(emotion_classes, emotion_embedding_dim)
        # self.emotion_embedding = nn.Parameter(torch.randn(emotion_classes, emotion_embedding_dim))
        self.input_projection = nn.Linear(mel_dim + pitch_dim + energy_dim, fft_dim)

        self.trans_enc = nn.TransformerEncoderLayer(d_model=fft_dim, nhead=num_heads, dim_feedforward=fft_dim, batch_first=True)
        self.conv1d = nn.Conv1d(fft_dim, fft_dim, kernel_size, padding=kernel_size // 2)
        
        self.emotion_prediction = nn.Linear(fft_dim, 5)
        self.feature_projection = nn.Linear(fft_dim, fft_dim)

    def forward(self, mel, pitch=None, energy=None):
        if pitch is None or energy is None:
            x = mel
        else:
            x = torch.cat([mel, pitch, energy], dim=-1)
        x = self.input_projection(x)
        
        x = self.trans_enc(x)
        x = x.transpose(1, 2)  # Conv1D expects (batch, channels, length)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # Switch back to (batch, length, channels)
        
        # emotion_embed = self.emotion_embedding(emotion_class).expand(-1, x.size(1), -1)
        # x = x + emotion_embed
        # emotion_prediction = torch.matmul(x, self.emotion_embedding.T)
        # intensity_representation = torch.matmul(torch.nn.functional.softmax(emotion_prediction, dim=1), self.emotion_embedding)
        emotion_prediction = self.emotion_prediction(x)
        x = self.feature_projection(x)
        
        return x, emotion_prediction

# class IntensityExtractor(nn.Module):
#     def __init__(self, mel_dim, pitch_dim, energy_dim, emotion_embedding_dim, fft_dim, num_heads, num_layers, kernel_size=1):
#         super(IntensityExtractor, self).__init__()
#         self.conv_net = Conv_Net(
#             channels=[mel_dim, 64, 128, 128, 256, emotion_embedding_dim], 
#             conv_kernels=[3, 3, 3, 3, 3], conv_strides=[2, 1, 2, 1, 2], 
#             dropout=0.1
#         )
#         self.dropout = nn.Dropout(0.1)
#         self.attn_head = AdditiveAttention(
#             dropout=0.1,
#             query_vector_dim=emotion_embedding_dim,
#             candidate_vector_dim=emotion_embedding_dim)
#         self.emotion_embeddings = nn.Parameter(
#             torch.randn(5, emotion_embedding_dim)
#         )

#     def forward(self, mel, pitch=None, energy=None):
#         intensity_representation = self.conv_net(mel)
#         ref_enc_out = self.attn_head(intensity_representation)
#         ref_enc_out = self.dropout(ref_enc_out)
#         attention_weight = torch.matmul(ref_enc_out, self.emotion_embeddings.T)
#         return intensity_representation, attention_weight





class RankModel(nn.Module):
    def __init__(self):
        super(RankModel, self).__init__()
        self.intensity_extractor = IntensityExtractor(
            num_heads=4,
            num_layers=4,
            kernel_size=5,
            energy_dim=0,
            pitch_dim=0,
            mel_dim=80,
            fft_dim=128,
            emotion_embedding_dim=128,
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.alpha = 0.4
        self.beta = 0.6

    def forward(self, x):
        _i, i = self.intensity_extractor(x)
        
        h = i.mean(dim=1)
        
        r = self.projector(_i.mean(dim=1))
        
        return (
            _i,  # Intensity representations
            h,  # Mean intensity representations
            r,  # Intensity predictions
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
    print(create_model('rank'))

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