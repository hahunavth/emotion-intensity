import os
import pickle as pk
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import random


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
            
        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
                            )
    
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=False).to(inputs.device) #.cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        mask.requires_grad_(True)

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class StrengthNet(nn.Module):
    """
    Reimplementation of the StrengthNet model in PyTorch.
    See: https://github.com/ttslr/StrengthNet/blob/main/model.py
    """

    def __init__(self, n_classes=4):
        super(StrengthNet, self).__init__()
        self.n_classes = n_classes

        self.fc_input = nn.Linear(80, 257)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.blstm1 = nn.LSTM(
            128 * 4,
            128,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
            #             num_layers=2,
        )
        self.blstm2 = nn.LSTM(
            128 * 4,
            128,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
            # num_layers=3,
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.n_classes),
        )
            # nn.Linear(256, self.n_classes),
        self.dropout = nn.Dropout(0.3)
        self.frame_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
            # nn.Linear(128, 1),
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=1)

        self.attn = Attention(128 * 2, batch_first=True)

        # apply weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_lens):
        """
        x: (batch_size, lengh, 80)
        """
        batch_size = len(x_lens)
        x = F.relu(self.fc_input(x))
        x = x.view(batch_size, 1, -1, 257)  # (N, C, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.transpose(1, 2)  # (N, 128, L, 4)
        x = x.reshape(batch_size, -1, 128 * 4)  # (bs, l, 128*4) # (N, L, H)

        x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        # BLSTM1
        out, (hn, cn) = self.blstm1(x)
        out, input_sizes = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)

        # BLSTM2
        out2, (hn, cn) = self.blstm2(x)
        out2, input_sizes = nn.utils.rnn.pad_packed_sequence(out2, batch_first=True)
        # get last output by x_lens
        # out2 = torch.stack(
        #     [out2[i, x_lens[i] - 1, :] for i in range(batch_size)]
        # )  # (bs, 256)
        # out2 = self.dropout(out2)
        # out2 = self.fc2(out2)
        #         out2 = self.softmax(out2)

        attn_out, attns = self.attn(out2, x_lens)
        if attn_out.ndim == 1:
            attn_out = attn_out.unsqueeze(0)
        emo_pred = self.fc2(attn_out)

        # Frame Score
        frame_score = self.frame_fc(out)

        # Average Score through frames with x_lens
        x_lens = x_lens.to(frame_score.device)
        mask = torch.arange(frame_score.size(1), device=frame_score.device)[None, :] < x_lens[:, None]
        masked_frame_score = frame_score.masked_fill(~mask[:, :, None], 0)
        seq_sum = masked_frame_score.sum(dim=1)
        seq_count = mask.sum(dim=1, dtype=torch.float32)
        avg_score = seq_sum / seq_count[:, None]
        avg_score = nn.Sigmoid()(avg_score)
        # avg_score = self.avgpool(frame_score.permute(0, 2, 1)).squeeze(2)

        return (
            avg_score.squeeze(1),
            frame_score.squeeze(2),
            emo_pred,
        )  # (bs, 1), (bs, l, 1), (bs, n_classes)


class StrengthNetLoss(nn.Module):
    def __init__(self):
        super(StrengthNetLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, pred, target):
        avg_score, frame_score, one_hot = pred
        target_score, target_emotion_id = target

        cat_loss = self.ce_loss(one_hot, target_emotion_id)
        avg_loss = self.mae_loss(avg_score, target_score)
        # frame_loss = self.mae_loss(frame_score.squeeze(-1), target_score)

        total_loss = cat_loss + avg_loss  # + frame_loss

        losses = {
            "cat_loss": cat_loss.item(),
            "avg_loss": avg_loss.item(),
            # "frame_loss": frame_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, losses


def _mask_spec_augment(
    spec: np.ndarray,
    num_mask=2,
    freq_masking_max_percentage=0.1,
    time_masking_max_percentage=0.1,
):
    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0 : f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0 : t0 + num_frames_to_mask, :] = 0

    return spec


def _resize_spec_aug(mel, height):  # 68-92
    """
    See FreeVC repo
    https://github.com/OlaWod/FreeVC/blob/main/utils.py?fbclid=IwAR2fgXMm1CWG-p_X61ISYud7MS4I12EnqRqlSYK_Ns3gJn3l8SZoGs0w1vs#L52
    """
    # r = np.random.random()
    # rate = r * 0.3 + 0.85 # 0.85-1.15
    # height = int(mel.size(-2) * rate)
    tgt = torchvision.transforms.functional.resize(mel, (height, mel.size(-1)))
    if height >= mel.size(-2):
        return tgt[:, : mel.size(-2), :]
    else:
        silence = tgt[:, -1:, :].repeat(1, mel.size(-2) - height, 1)
        silence += torch.randn_like(silence) / 10
        return torch.cat((tgt, silence), 1)


class EsdStrengthDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["path"]
        score = self.df.iloc[idx]["score"]
        emotion = self.df.iloc[idx]["emotion"]
        emotion_id = self.df.iloc[idx]["emotion_id"]
        mel_path = self.df.iloc[idx]["mel_path"]

        with open(mel_path, "rb") as f:
            mel = pk.load(f)

        return {
            "path": path,
            "score": score,
            "emotion": emotion,
            "emotion_id": emotion_id,
            "mel": mel,
        }

    @staticmethod
    def get_mel_path_from_audio_path(audio_path, preprocessed_basedir):
        basename = (
            audio_path.split("/")[-1].replace(".wav", "")
            + "_"
            + audio_path.split("/")[-2].lower()
        )
        return f"{preprocessed_basedir}/mel/mel_{basename}.pkl"

    @classmethod
    def from_csv(cls, path, emotion2idx, preprocessed_basedir, val_speakers, **kwargs):
        df = pd.read_csv(path)
        df.columns = ["path", "score"]

        # drop rows without mel file
        df["mel_path"] = df["path"].apply(
            lambda x: cls.get_mel_path_from_audio_path(x, preprocessed_basedir)
        )
        df = df[df["mel_path"].apply(lambda x: os.path.exists(x))]

        df["emotion"] = df["path"].apply(lambda x: x.split("/")[-2].lower())
        df["emotion_id"] = df["emotion"].apply(lambda x: emotion2idx[x])

        df["speaker"] = df["path"].apply(lambda x: x.split("/")[-3])

        print("Loaded dataset with", len(df), "samples")

        # train test split
        val_df = df[df["speaker"].isin(val_speakers)]
        train_df = df[~df["speaker"].isin(val_speakers)]
        print("Train samples:", len(train_df))
        print("Val samples:", len(val_df))

        # reset index
        val_df = val_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        return cls(train_df, **kwargs), cls(val_df, **kwargs)

    @staticmethod
    def collate_fn(batch):
        # padding mels
        mel = []
        mel_lens = []
        for x in batch:
            mel.append(
                torch.from_numpy(
                    _mask_spec_augment(x["mel"]),
                ).float()
            )
            mel_lens.append(len(x["mel"]))
        
        mel_lens = torch.LongTensor(mel_lens)
        mel_lens, perm_idx = mel_lens.sort(0, descending=True)
        
        mel = nn.utils.rnn.pad_sequence(mel, batch_first=True)
        score = torch.tensor([x["score"] for x in batch], dtype=torch.float32)
        emotion_id = torch.tensor([x["emotion_id"] for x in batch], dtype=torch.long)
        
        mel = mel[perm_idx]
        score = score[perm_idx]
        emotion_id = emotion_id[perm_idx]
        
        # mel = nn.utils.rnn.pack_padded_sequence(mel, mel_lens, batch_first=True, enforce_sorted=False)
        
        return {
            "mel": mel,
            "score": score,
            "emotion_id": emotion_id,
            "mel_lens": mel_lens,
        }


import numpy as np
import torch
import math
from torch.functional import F


def mixup_data(x_emo, x_neu, alpha=1.0, lam=None):
    """Applies mixup augmentation to the data.

    Args:
    x_emo (Tensor): Input data (e.g., features of speech samples).
    x_neu (Tensor): Input data (e.g., features of speech samples).

    Returns:
    mixed_x (Tensor): The mixed input data.
    lam (float): The mixup coefficient.
    """
    if lam is None:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

    x_mixed = lam * x_emo + (1 - lam) * x_neu

    return x_mixed, lam


def mixup_criterion(pred, y_emo, y_neu, lam) -> torch.Tensor:
    w_emo = math.sqrt(16)
    w_neu = math.sqrt(1)
    l_emo = sum(
        [F.cross_entropy(pred[i], y_emo[i]) * lam[i] for i in range(len(pred))]
    ) / len(pred)
    l_neu = sum(
        [F.cross_entropy(pred[i], y_neu[i]) * (1 - lam[i]) for i in range(len(pred))]
    ) / len(pred)

    loss = (w_emo * l_emo + w_neu * l_neu) / (w_emo + w_neu)
    return loss


def rank_loss(ri, rj, lam_diff) -> torch.Tensor:
    p_hat_ij = F.sigmoid(ri - rj)
    rank_loss = torch.mean(
        -lam_diff @ torch.log(p_hat_ij) - (1 - lam_diff) @ torch.log(1 - p_hat_ij)
    ) / lam_diff.size(0)
    return rank_loss


from torch import nn


class StrengthNetMixupLoss(nn.Module):
    alpha = 1.0
    beta = 0.1

    def forward(self, predi, predj, y_emo, y_neu, lam_i, lam_j):
        _, hi, ri = predi
        _, hj, rj = predj
        lam_diff = (lam_i - lam_j) / 2 + 0.5

        losses = {}
        mixup_loss_i = mixup_criterion(hi, y_emo, y_neu, lam_i)
        mixup_loss_j = mixup_criterion(hj, y_emo, y_neu, lam_j)
        mixup_loss = mixup_loss_i + mixup_loss_j
        losses.update(
            {
                "mi": mixup_loss_i.item(),
                "mj": mixup_loss_j.item(),
            }
        )

        ranking_loss = rank_loss(ri, rj, lam_diff)
        losses.update(
            {
                "rank": ranking_loss.item(),
            }
        )

        total_loss = self.alpha * mixup_loss + self.beta * ranking_loss
        losses.update(
            {
                "total": total_loss.item(),
            }
        )

        return total_loss, losses


import random


class RankMixupDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, emo_ids, neu_ids, alpha=1.0, validation=False):
        self.dataset = dataset
        self.alpha = alpha
        self.emo_ids = emo_ids
        self.neu_ids = neu_ids
        self.validation = validation

    def __len__(self):
        if self.validation is False:
            return len(self.neu_ids)
        else:
            return max(len(self.emo_ids), len(self.neu_ids))

    def __getitem__(self, idx):
        if self.validation is False:
            sample_emo = self.dataset[
                self.emo_ids[np.random.randint(0, len(self.emo_ids))]
            ]
            sample_neu = self.dataset[self.neu_ids[idx % len(self.neu_ids)]]
        else:
            sample_emo = self.dataset[self.emo_ids[idx % len(self.emo_ids)]]
            sample_neu = self.dataset[self.neu_ids[idx % len(self.neu_ids)]]

        x_emo, y_emo = sample_emo["mel"], sample_emo["emotion_id"]
        x_neu, y_neu = sample_neu["mel"], sample_neu["emotion_id"]

        x_emo = torch.from_numpy(x_emo).float()
        x_neu = torch.from_numpy(x_neu).float()
        y_emo = torch.tensor(y_emo, requires_grad=False)
        y_neu = torch.tensor(y_neu, requires_grad=False)

        return x_emo, x_neu, y_emo, y_neu

    def collate_fn(self, batch):
        mel_emo, mel_neu, y_emo, y_neu = zip(*batch)
        batch_size = len(mel_neu)

        lam_i = np.random.beta(self.alpha, self.alpha) if self.alpha != 0 else 0
        lam_j = np.random.beta(self.alpha, self.alpha) if self.alpha != 0 else 0
        # if lam_i > 0.5:
        #     lam_i = 1 - lam_i
        # if lam_j < 0.5:
        #     lam_j = 1 - lam_j
        lam_i = [lam_i] * batch_size
        lam_j = [lam_j] * batch_size

        xis = []
        xjs = []
        xi_lens = []
        xj_lens = []
        for i in range(batch_size):
            # if self.rand_lam_per_batch is False:
            _lam_i = np.random.beta(self.alpha, self.alpha) if self.alpha != 0 else 0
            _lam_j = np.random.beta(self.alpha, self.alpha) if self.alpha != 0 else 0
            # if lam_i > 0.5:
            #     lam_i = 1 - lam_i
            # if lam_j < 0.5:
            #     lam_j = 1 - lam_j
            if self.validation:
                _lam_i = 0.
                _lam_j = 1.
            lam_i[i] = _lam_i
            lam_j[i] = _lam_j

            neu_len = mel_neu[i].shape[0]
            emo_len = mel_emo[i].shape[0]
            min_mel_len = min(neu_len, emo_len)
            neu_start = random.randint(0, neu_len - min_mel_len)
            emo_start = random.randint(0, emo_len - min_mel_len)
            neu_mel = mel_neu[i][neu_start : neu_start + min_mel_len]
            emo_mel = mel_emo[i][emo_start : emo_start + min_mel_len]
            xi, _ = mixup_data(emo_mel, neu_mel, lam_i[i])
            xj, _ = mixup_data(emo_mel, neu_mel, lam_j[i])
            xis.append(xi)
            xjs.append(xj)
        xi_lens = [len(x) for x in xis]
        xj_lens = [len(x) for x in xjs]
        max_xi_len = max(xi_lens)
        max_xj_len = max(xj_lens)
        xi = [F.pad(x, (0, 0, 0, max_xi_len - len(x))) for x in xis]
        xj = [F.pad(x, (0, 0, 0, max_xj_len - len(x))) for x in xjs]
        xi = torch.stack(xi)
        xj = torch.stack(xj)
        y_emo = torch.tensor(y_emo, requires_grad=False)
        y_neu = torch.tensor(y_neu, requires_grad=False)
        lam_i = torch.tensor(lam_i, requires_grad=False)
        lam_j = torch.tensor(lam_j, requires_grad=False)
        xi_lens = torch.tensor(xi_lens, requires_grad=False)
        xj_lens = torch.tensor(xj_lens, requires_grad=False)

        return xi, xj, y_emo, y_neu, lam_i, lam_j, xi_lens, xj_lens


class AvgMeter:
    def __init__(self):
        self.losses = {}

    def update(self, losses):
        for k, v in losses.items():
            if k not in self.losses:
                self.losses[k] = []
            self.losses[k].append(v)

    def get_avg(self):
        return {k: sum(v) / len(v) for k, v in self.losses.items()}

    def reset(self):
        self.losses = {}


# training

e2id = {
    "angry": 0,
    "happy": 1,
    "neutral": 2,
    "sad": 3,
    "surprise": 4,
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StrengthNet(n_classes=len(e2id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_ds, val_ds = EsdStrengthDataset.from_csv(
        "./datasets/ESD_score_list_all.csv",
        e2id,
        preprocessed_basedir="./datasets/esd_processed",
        val_speakers={"0001", "0005", "0011", "0015"},
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=EsdStrengthDataset.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=32, shuffle=False, collate_fn=EsdStrengthDataset.collate_fn
    )
    criterion = StrengthNetLoss()


    # for epoch in range(10):
    #     train_meter = AvgMeter()
    #     val_meter = AvgMeter()

    #     model.train()
    #     for batch in train_loader:
    #         x = batch["mel"].to(device)
    #         x_lens = batch["mel_lens"]
    #         y = (batch["score"].to(device), batch["emotion_id"].to(device))

    #         pred = model(x, x_lens)
    #         loss, losses = criterion(pred, y)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         # print(losses)
    #         train_meter.update(losses)

    #     print(f"Epoch {epoch} train loss: ", train_meter.get_avg())

    #     model.eval()
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             x = batch["mel"].to(device)
    #             x_lens = batch["mel_lens"]
    #             y = (batch["score"].to(device), batch["emotion_id"].to(device))
    #             pred = model(x, x_lens)
    #             loss, losses = criterion(pred, y)
    #             # print(losses)
    #             val_meter.update(losses)
    #         print(f"Epoch {epoch} val loss: ", val_meter.get_avg())

    #     #
    #     emo_ids = val_ds.df[val_ds.df["emotion"] != "neutral"].index
    #     neu_ids = val_ds.df[val_ds.df["emotion"] == "neutral"].index
    #     mix_val_ds = RankMixupDataset(val_ds, emo_ids, neu_ids, alpha=0.0, validation=True)
    #     val_loader2 = torch.utils.data.DataLoader(
    #         mix_val_ds, batch_size=32, shuffle=False, collate_fn=mix_val_ds.collate_fn
    #     )
    #     emo_lb = []
    #     emo_pred = []
    #     emo_neu_pred = []
    #     rank_true = []
    #     for idx, batch in enumerate(val_loader2):
    #         model.eval()
    #         xi, xj, y_emo, y_neu, lam_i, lam_j, xi_lens, xj_lens = batch

    #         xi = xi.to(device)
    #         xj = xj.to(device)
    #         lam_i = lam_i.to(device)
    #         lam_j = lam_j.to(device)
    #         y_neu = y_neu.to(device)
    #         y_emo = y_emo.to(device)

    #         ri, _, hi = model(xi, xi_lens)
    #         rj, _, hj = model(xj, xj_lens)

    #         #         loss, losses = criterion((_, hi, ri), (_, hj, rj), y_emo, y_neu, lam_i, lam_j)

    #         y_neu_pred = F.softmax(hi, dim=1).argmax(dim=1)
    #         y_pred = F.softmax(hj, dim=1).argmax(dim=1)

    #         emo_lb.append(y_emo.detach().cpu())
    #         emo_pred.append(y_pred.detach().cpu())
    #         emo_neu_pred.append(y_neu_pred.detach().cpu())

    #         rank_true.append((ri < rj).detach().cpu())

    #     #         val_meter.update(losses)
    #     #     print("Val loss: ", val_meter.get_avg())

    #     emo_lb = torch.cat(emo_lb, dim=0)
    #     emo_pred = torch.cat(emo_pred, dim=0)
    #     emo_neu_pred = torch.cat(emo_neu_pred, dim=0)
    #     rank_true = torch.cat(rank_true, dim=0)
    #     from sklearn.metrics import accuracy_score, f1_score

    #     emo_acc = accuracy_score(emo_lb, emo_pred)
    #     print("Emotion Accuracy: ", emo_acc)
    #     emo_neu_acc = accuracy_score(emo_neu_pred, torch.zeros_like(emo_neu_pred))
    #     print("Neutral Accuracy: ", emo_neu_acc)

    #     print("Rank Accuracy: ", accuracy_score(rank_true, torch.ones_like(rank_true)))

    #     print()

    # =====================
        
    emo_ids = train_ds.df[train_ds.df["emotion"] != "neutral"].index
    neu_ids = train_ds.df[train_ds.df["emotion"] == "neutral"].index
    mix_train_ds = RankMixupDataset(train_ds, emo_ids, neu_ids, alpha=0.8)

    emo_ids = val_ds.df[val_ds.df["emotion"] != "neutral"].index
    neu_ids = val_ds.df[val_ds.df["emotion"] == "neutral"].index
    mix_val_ds = RankMixupDataset(val_ds, emo_ids, neu_ids, alpha=0., validation=True)

    train_loader = torch.utils.data.DataLoader(
        mix_train_ds, batch_size=32, shuffle=True, collate_fn=mix_train_ds.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        mix_val_ds, batch_size=32, shuffle=False, collate_fn=mix_val_ds.collate_fn
    )

    criterion = StrengthNetMixupLoss()


    for epoch in range(50):
        train_meter = AvgMeter()
        val_meter = AvgMeter()
        for idx, batch in enumerate(train_loader):
            model.train()
            xi, xj, y_emo, y_neu, lam_i, lam_j, xi_lens, xj_lens = batch

            xi = xi.to(device)
            xj = xj.to(device)
            lam_i = lam_i.to(device)
            lam_j = lam_j.to(device)
            y_neu = y_neu.to(device)
            y_emo = y_emo.to(device)

            ri, _, hi = model(xi, xi_lens)
            rj, _, hj = model(xj, xj_lens)

            loss, losses = criterion((_, hi, ri), (_, hj, rj), y_emo, y_neu, lam_i, lam_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_meter.update(losses)
        print("Train loss: ", train_meter.get_avg())

        emo_lb = []
        emo_pred = []
        emo_neu_pred = []
        rank_true = []
        for idx, batch in enumerate(val_loader):
            model.eval()
            xi, xj, y_emo, y_neu, lam_i, lam_j, xi_lens, xj_lens = batch

            xi = xi.to(device)
            xj = xj.to(device)
            lam_i = lam_i.to(device)
            lam_j = lam_j.to(device)
            y_neu = y_neu.to(device)
            y_emo = y_emo.to(device)

            ri, _, hi = model(xi, xi_lens)
            rj, _, hj = model(xj, xj_lens)

            loss, losses = criterion((_, hi, ri), (_, hj, rj), y_emo, y_neu, lam_i, lam_j)

            y_neu_pred = F.softmax(hi, dim=1).argmax(dim=1)
            y_pred = F.softmax(hj, dim=1).argmax(dim=1)

            emo_lb.append(y_emo.detach().cpu())
            emo_pred.append(y_pred.detach().cpu())
            emo_neu_pred.append(y_neu_pred.detach().cpu())

            rank_true.append((ri < rj).detach().cpu())

            val_meter.update(losses)
        print("Val loss: ", val_meter.get_avg())

        emo_lb = torch.cat(emo_lb, dim=0)
        emo_pred = torch.cat(emo_pred, dim=0)
        emo_neu_pred = torch.cat(emo_neu_pred, dim=0)
        rank_true = torch.cat(rank_true, dim=0)
        from sklearn.metrics import accuracy_score, f1_score
        emo_acc = accuracy_score(emo_lb, emo_pred)
        print("Emotion Accuracy: ", emo_acc)
        emo_neu_acc = accuracy_score(emo_neu_pred, torch.zeros_like(emo_neu_pred).fill_(2))
        print("Neutral Accuracy: ", emo_neu_acc)

        print("Rank Accuracy: ", accuracy_score(rank_true, torch.ones_like(rank_true)))

        print()
        
        # save
        torch.save(model.state_dict(), f"strengthnet_ep{epoch}.pth")

    # save model
    torch.save(model.state_dict(), "strengthnet.pth")

    # if __name__ == "__main__":
    # model = StrengthNet()
    # print("N of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # x = torch.randn(32, 129, 80)
    # x = torch.randn(32, 80, 120)
    # avg_score, frame_score, out2 = model(x)
    # print(avg_score.size(), frame_score.size(), out2.size())
