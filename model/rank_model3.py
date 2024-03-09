from collections import OrderedDict
from turtle import forward

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class IntensityExtractor(torch.nn.Module):
    def __init__(
        self,
        mel_dim=80,
        pitch_dim=0,
        energy_dim=0,
        fft_dim=256,
        num_heads=4,
        kernel_size=3,
        n_emotion=5,
        emotion_dim=32,
        n_layers=3,
    ):
        super(IntensityExtractor, self).__init__()
        self.input_projection = nn.Linear(mel_dim + pitch_dim + energy_dim, fft_dim)
        d_k = d_v = fft_dim // num_heads

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    fft_dim,
                    num_heads,
                    d_k,
                    d_v,
                    d_inner=1024,
                    kernel_size=[9, 1],
                    dropout=0.2,
                )
                for _ in range(n_layers)
            ]
        )

        self.emotion_embedding = nn.Embedding(n_emotion - 1, emotion_dim)
        
        self.emo_prediction = nn.Linear(fft_dim, n_emotion)

    def forward(self, mel, mel_lens, pitch=None, energy=None, emo_id=None):
        x = mel  # (batch, length, channels)
        x = self.input_projection(x)

        mask = torch.arange(x.size(1)).unsqueeze(0).to(mel.device) >= mel_lens.unsqueeze(1)
        slf_attn_mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)
        
        for layer in self.layer_stack:
            x, _ = layer(x, mask=mask, slf_attn_mask=slf_attn_mask)
            
        if emo_id is not None:
            emotion_embed = (
                self.emotion_embedding(emo_id - 1)
                .unsqueeze(1)
                .expand(-1, x.size(1), -1)
            )
            emotion_embed = F.pad(emotion_embed, (0, x.size(2) - emotion_embed.size(2)))
            x = x + emotion_embed
            
        i = self.emo_prediction(x)
        
        return i, x


class RankModel(nn.Module):
    def __init__(
        self, 
        fft_dim=256, 
        n_emotion=5,
    ):
        super(RankModel, self).__init__()
        self.fft_dim = fft_dim
        self.n_emotion = n_emotion
        
        self.intensity_extractor = IntensityExtractor(fft_dim=fft_dim, n_emotion=n_emotion)
        # self.projector = nn.Linear(fft_dim, 1)
        self.projector = nn.Linear(n_emotion, 1)

    def forward(self, mel, mel_lens, pitch=None, energy=None, emo_id=None):
        if isinstance(mel_lens, list):
            mel_lens = torch.tensor(mel_lens).to(mel.device)
        
        i, x = self.intensity_extractor(mel, mel_lens, pitch=pitch, energy=energy, emo_id=emo_id)
        
        mask = torch.arange(x.size(1)).unsqueeze(0).to(mel.device) >= mel_lens.unsqueeze(1)

        h = i.masked_fill(mask.unsqueeze(-1), 0)
        h = i.sum(dim=1) / mel_lens.unsqueeze(1)

        # r = self.projector(x)
        r = self.projector(h).squeeze(0)
        r = F.tanh(r)
        # r = r.squeeze(2).masked_fill(mask, 0)
        # r = r.sum(dim=1) / mel_lens.unsqueeze(1)
        
        return i, h, r


if __name__ == "__main__":
    # # Test
    # fft_block = FFTBlock(80, 4, 20, 20, 256, [3, 3])
    # x = torch.randn(10, 100, 80)
    # mask = torch.zeros(10, 100)
    # mask[:, 10:] = 1
    # mask = mask.bool()
    # slf_attn_mask = mask.unsqueeze(1).expand(-1, 100, -1)
    # output, attn = fft_block(x, mask=mask, slf_attn_mask=slf_attn_mask)
    # print(output.shape, attn.shape)
    # # print(output)
    # # print(attn)
    # print("FFTBlock test passed")
    
    # ie = IntensityExtractor()
    # mel = torch.randn(10, 100, 80)
    # mel_lens = torch.randint(80, 100, (10,))
    # emo_id = torch.randint(1, 5, (10,))
    # i, x = ie(mel, mel_lens, emo_id=emo_id)
    # print(i.shape, x.shape)
    
    rm = RankModel()
    mel = torch.randn(10, 100, 80)
    mel_lens = torch.randint(80, 100, (10,))
    emo_id = torch.randint(1, 5, (10,))
    i, h, r = rm(mel, mel_lens, emo_id=emo_id)
    print(i.shape, h.shape, r.shape)
    print("RankModel test passed")