import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math
import json


class SpatialDropout1D(nn.Module):
    def __init__(self, drop_rate):
        super(SpatialDropout1D, self).__init__()

        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        inputs = self.dropout(inputs.unsqueeze(2)).squeeze(2)
        inputs = inputs.permute(0, 2, 1)

        return inputs


class Conv_Net(
    nn.Module,
):
    def __init__(
        self,
        channels=[80, 128],
        conv_kernels=[
            3,
        ],
        conv_strides=[
            1,
        ],
        maxpool_kernels=[
            2,
        ],
        maxpool_strides=[
            2,
        ],
        dropout=0.2,
    ):

        super(Conv_Net, self).__init__()

        convs = []
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.maxpool_kernels = maxpool_kernels
        self.maxpool_strides = maxpool_strides

        for i, (in_channels, out_channels) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            conv = [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.conv_kernels[i],
                    stride=self.conv_strides[i],
                    padding=self.conv_kernels[i] - 1,
                ),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(),
            ]
            convs += conv

        self.conv_net = nn.ModuleList(convs)
        self.dropout = SpatialDropout1D(dropout)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length + 2 * (kernel_size - 1) - kernel_size) // stride + 1

        def _maxpool_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for index in range(len(self.conv_kernels)):
            input_lengths = _conv_out_length(
                input_lengths, self.conv_kernels[index], self.conv_strides[index]
            )
            # input_lengths = _maxpool_out_length(input_lengths, self.maxpool_kernels[index], self.maxpool_strides[index])

        return input_lengths.to(torch.long)

    def _get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = (
            torch.arange(0, max_len, device=lengths.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask

        # inputs_lengths = torch.sum(~masks, dim=-1)
        # outputs_lengths = self._get_feat_extract_output_lengths(inputs_lengths)
        # masks = self._get_mask_from_lengths(outputs_lengths)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        for conv in self.conv_net:
            inputs = conv(inputs)
        inputs = self.dropout(inputs)
        inputs = inputs.permute(0, 2, 1)

        return inputs


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


class Reference_Encoder(nn.Module):
    def __init__(
        self,
        in_channels=80,
        dropout=0.2,
    ) -> None:
        super(Reference_Encoder, self).__init__()

        self.conv_net = Conv_Net(
            channels=[in_channels, 64, 128, 128, 256, 128],
            conv_kernels=[3, 3, 3, 3, 3],
            conv_strides=[2, 1, 2, 1, 2],
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        self.attn_head = AdditiveAttention(
            dropout=dropout, query_vector_dim=128, candidate_vector_dim=128
        )

    def forward(self, inputs):
        """
        inputs: (batch, length, channels)
        """
        outputs = self.conv_net(inputs)
        outputs = self.attn_head(outputs)
        outputs = self.dropout(outputs)

        return outputs


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
        pitch_dim=16,
        energy_dim=16,
        fft_dim=128,
        num_heads=8,
        kernel_size=1,
        n_emotion=5,
        n_bins=10,
        stats_path="datasets/esd_processed/stats.json",
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
            torch.exp(torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)),
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

        self.input_projection = nn.Linear(mel_dim, fft_dim) #(mel_dim + pitch_dim + energy_dim, fft_dim)



    def forward(self, mel, pitch=None, energy=None, emo_id=None):
        x = mel
        # if pitch is None or energy is None:
        #     # x = mel  # (batch, length, channels)
        #     raise ValueError("pitch and energy must be provided")
        # else:
        #     p_emb = self.pitch_embedding(torch.bucketize(pitch, self.pitch_bins))
        #     e_emb = self.energy_embedding(torch.bucketize(energy, self.energy_bins))
        #     x = torch.cat([mel, p_emb, e_emb], dim=2)  # (batch, length, mel+pitch+energy channels)

        x = self.input_projection(x)  # (batch, length, n_chanel)


        return None, x  # (batch, length, n_emotion), (batch, length, fft_dim)


class RankModel(nn.Module):
    def __init__(self, fft_dim=128, n_emotion=5, **kwargs):
        super(RankModel, self).__init__()
        self.intensity_extractor = IntensityExtractor(fft_dim=fft_dim, n_emotion=n_emotion, **kwargs)
        self.pos_enc = PositionalEncoder(fft_dim, 1000)
        self.ref_enc = Reference_Encoder(in_channels=fft_dim)
        self.emotion_embedding = nn.Embedding(n_emotion - 1, fft_dim // 2)
        
        self.ref_feat_fc = nn.Sequential(
            nn.Linear(fft_dim, fft_dim * 2),
            nn.BatchNorm1d(fft_dim * 2),
            nn.ReLU(),
            nn.Linear(fft_dim * 2, fft_dim * 2),
            nn.BatchNorm1d(fft_dim * 2),
            nn.ReLU(),
            nn.Linear(fft_dim * 2, fft_dim),
            nn.BatchNorm1d(fft_dim),
            nn.Linear(fft_dim, fft_dim // 2),
            nn.BatchNorm1d(fft_dim // 2),
            nn.Linear(fft_dim // 2, fft_dim),
            nn.BatchNorm1d(fft_dim),
            nn.ReLU(),
        )
        self.emotion_predictor = nn.Linear(fft_dim, n_emotion)
        self.rank_predictor = nn.Linear(fft_dim, 1)

    def forward(self, mel, pitch, energy, mel2, pitch2, energy2, emo_id, lam, lam2):
        _, x = self.intensity_extractor(mel, pitch, energy, emo_id)
        _, x2 = self.intensity_extractor(mel2, pitch2, energy2, emo_id)
        
        x = self.pos_enc(x)  # (batch, length, n_chanel)
        x2 = self.pos_enc(x2)  # (batch, length, n_chanel)
        x = self.ref_enc(x)  # (batch, 128)
        x2 = self.ref_enc(x2)  # (batch, 128)
        
        if lam is not None:
            if isinstance(lam, list):
                lam = torch.tensor(lam)
            # NOTE: mainfold mixup
            x = torch.cat([(lam[i] * x2[i] + (1-lam)[i] * x[i]).unsqueeze(0) for i in range(len(x))])
            x2 = torch.cat([(lam2[i] * x2[i] + (1-lam2)[i] * x[i]).unsqueeze(0) for i in range(len(x))])
            # x = lam * x + (1 - lam) * x2
            # x2 = lam * x2 + (1 - lam) * x

        if emo_id is not None:
            emotion_embed = self.emotion_embedding(emo_id - 1)
            # padding to same size
            emotion_embed = F.pad(emotion_embed, (0, x.size(-1) - emotion_embed.size(-1)))
            x = x + emotion_embed
            x2 = x2 + emotion_embed

        x = self.ref_feat_fc(x)
        x2 = self.ref_feat_fc(x2)

        h = self.emotion_predictor(x)
        h2 = self.emotion_predictor(x2)
        r = self.rank_predictor(x)
        r2 = self.rank_predictor(x2)

        return (
            None,
            h,
            r,
            None,
            h2,
            r2,
        )


if __name__ == "__main__":
    # ref_enc = Reference_Encoder()
    # mel = torch.randn(1, 160, 80)
    # out = ref_enc(mel)
    # print(out.shape)

    # intensity_extractor = IntensityExtractor()
    # mel = torch.randn(1, 160, 80)
    # pitch = torch.randn(1, 160)
    # energy = torch.randn(1, 160)
    # emo_id = torch.tensor([1])
    # i, x = intensity_extractor(mel, pitch, energy, emo_id)

    rank_model = RankModel()
    mel = torch.randn(1, 160, 80)
    pitch = torch.randn(1, 160)
    energy = torch.randn(1, 160)
    emo_id = torch.tensor([1])
    _, h, r = rank_model(mel, pitch, energy, emo_id)
    print(h.shape, r.shape)
    print("N params: ", sum(p.numel() for p in rank_model.parameters() if p.requires_grad))
