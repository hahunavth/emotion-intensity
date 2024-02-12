
import json
import os

import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torch.nn.functional as F

from text import phoneme_to_ids
from utils import pad_1D, pad_2D


def mixup_data(x_emo, x_neu, alpha=1.0, lam=None):
    """Applies mixup augmentation to the data.

    Args:
    x_emo (Tensor): Input data (e.g., features of speech samples).
    x_neu (Tensor): Input data (e.g., features of speech samples).

    Returns:
    mixed_x (Tensor): The mixed input data.
    lam (float): The mixup coefficient.
    """
    if not lam:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

    x_mixed = lam * x_emo + (1 - lam) * x_neu
    return x_mixed, lam


from torch.utils.data import Dataset


class EmoFS2Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.label2index = preprocess_config["emotion2id"]
        self.num_label = len(self.label2index.items())

        self.basename, self.speaker, self.text, self.raw_text, self.ser_label = self.process_meta(
            filename
        )
        self.sort = sort
        self.drop_last = drop_last
        
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        # with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            # self.label2index = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        raw_text = self.raw_text[idx]
        emotion2id = self.label2index[self.ser_label[idx]]
        
        prefix = ""
        speaker = None
        speaker_id = None
        if self.speaker_map:
            speaker = self.speaker[idx]
            speaker_id = self.speaker_map[speaker]
            prefix = f"{speaker}-" 
        
        phone = np.array(phoneme_to_ids(self.text[idx]))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            prefix + "mel-{}.npy".format(basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            prefix + "pitch-{}.npy".format(basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            prefix + "energy-{}.npy".format(basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            prefix + "duration-{}.npy".format(basename),
        )
        duration = np.load(duration_path)
        if mel.shape[0] <= 8:
            return self.__getitem__(random.randint(0, self.__len__()-1))

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "emotion2id":emotion2id
        }
        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            spk = []
            text = []
            raw_text = []
            emo = []
            for line in f.readlines():
                items = line.strip("\n").split("|")
                if len(items) == 5:
                    n, s, t, r, e = items
                    spk.append(s)
                else:
                    n, t, r, e = items
                name.append(n)
                text.append(t)
                raw_text.append(r)
                emo.append(e)
            return name, spk, text, raw_text, emo
        
    def smooth_labels(self, labels, factor=0.1):
        labels *= (1 - factor)
        labels += (factor / labels.shape[0])
        return labels.astype(np.float32)
    
    def one_hot(self, a):
        return np.squeeze(np.eye(self.num_label)[a.reshape(-1)])
    
    def spec_augment(self, spec: np.ndarray, num_mask=2, 
                    freq_masking_max_percentage=0.1, time_masking_max_percentage=0.1):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        
        return spec
    
    def reprocess_for_test(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speaker_ids = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        emotion_labels = [data[idx]["emotion2id"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        
        speaker_ids = np.array(speaker_ids)
        emotion_labels = np.array(emotion_labels)
        emotion_labels = self.one_hot(emotion_labels).astype(np.float32)
        
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        
        return (
            ids,
            raw_texts,
            speaker_ids,
            texts,
            text_lens,
            max(text_lens),
            emotion_labels,
            mels,
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speaker_ids = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        emotion_labels = [data[idx]["emotion2id"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        
        speaker_ids = np.array(speaker_ids)
        emotion_labels = np.array(emotion_labels)
        emotion_labels = self.one_hot(emotion_labels)
        emotion_labels = self.smooth_labels(emotion_labels)
        
        reference_mels = [self.spec_augment(data[idx]["mel"]) for idx in idxs]
        
        reference_mels = pad_2D(reference_mels)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        
        return (
            ids,
            raw_texts,
            speaker_ids,
            texts,
            text_lens,
            max(text_lens),
            emotion_labels,
            mels,
            reference_mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
            
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        
        return output
    
    def collate_fn_for_test(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
            
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess_for_test(data, idx))
        
        return output


# def build_dict(ds_dir):
#     d = []
#     for i, dir in enumerate(os.listdir(ds_dir)):
#         base_dir = os.path.join(ds_dir, dir)
#         dic_file = os.path.join(ds_dir, dir, f"{dir}.txt")
#         if os.path.exists(dic_file):
#             d.append({
#                 "base_dir": base_dir,
#                 "dict_file": dic_file,
#             })
#     return d


# def get_list_files(info):
#     emotions = ['Happy', 'Sad', 'Surprise', 'Neutral', 'Angry']
#     list_files = []
#     for emotion in emotions:
#         list_files += [os.path.join(info['base_dir'], emotion, f) for f in os.listdir(os.path.join(info['base_dir'], emotion)) if f.endswith(".wav")]
#     return list_files


def get_list_filesss(ds_dir):
    # list_files = []

    # for info in build_dict(ds_dir):
    #     _list_files = get_list_files(info)
    #     list_files += _list_files
    # return list_files
    return os.listdir(ds_dir)


import audio as Audio
from torch.utils.data import Dataset
import torch
import pickle as pk


class ESDataset(Dataset):
    def __init__(self, list_files, base_dir=None):
        self.list_files = list_files
        self.base_dir = base_dir
        self.STFT = Audio.stft.TacotronSTFT(
            1024, # config["preprocessing"]["stft"]["filter_length"],
            256, # config["preprocessing"]["stft"]["hop_length"],
            1024, # config["preprocessing"]["stft"]["win_length"],
            80, # config["preprocessing"]["mel"]["n_mel_channels"],
            22050, # config["preprocessing"]["audio"]["sampling_rate"],
            0, # config["preprocessing"]["mel"]["mel_fmin"],
            8000 # config["preprocessing"]["mel"]["mel_fmax"],
        )
        self.emo2idx = {
            'neutral': 0,
            'happy': 1,
            'sad': 2,
            'angry': 3,
            'surprise': 4
        }

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.list_files[idx]
        emotion = file.replace(".pkl", "").split("_")[-1].lower()
        if self.base_dir is not None:
            file = os.path.join(self.base_dir, file)
        mel = pk.load(open(file, "rb"))
        pitch_file = file.replace("mel", "pitch")
        pitch = pk.load(open(pitch_file, "rb"))
        energy_file = file.replace("mel", "energy")
        energy = pk.load(open(energy_file, "rb"))
        
        return mel, pitch, energy, self.emo2idx[emotion], file


class MixDataset(Dataset):
    @classmethod
    def from_es_ds(cls, dataset: ESDataset, **kwargs):
        ser_label = [file.replace(".pkl", "").split("_")[-1].lower() for file in dataset.list_files]        

        emo_set = dataset.emo2idx.keys()
        emo_id_dict = {}
        
        for idx, emo in enumerate(ser_label):
            if emo not in emo_id_dict:
                emo_id_dict[emo] = []
            emo_id_dict[emo].append(idx)
        
        emo_set_without_neu = set(emo_set).difference(set(["neutral"]))
        
        def get_one_fn(idx):
            sample = dataset[idx]
            return {"mel": sample[0], "pitch": sample[1], "energy": sample[2], "emotion": sample[3]}
        return cls(get_one_fn, emo_id_dict, emo_set_without_neu=emo_set_without_neu, wrapped_ds=dataset, **kwargs)

    
    @classmethod
    def from_emofs2_ds(cls, dataset, **kwargs):
        emo_set = set(dataset.ser_label)
        emo_id_dict = {}
        
        emo_set_without_neu = set(dataset.ser_label).difference(set(["neutral"]))
        n_emo = len(emo_set)
        emo2id = {emo: i for i, emo in enumerate(emo_set)}
        for i, (emo, spk) in enumerate(zip(dataset.ser_label, dataset.speaker)):
            if spk == "neu":
                continue
            if emo not in emo_id_dict:
                emo_id_dict[emo] = []
            emo_id_dict[emo].append(i)
        def get_one_fn(idx):
            sample = dataset[idx]
            return {"mel": sample["mel"], "pitch": sample["pitch"], "energy": sample["energy"], "emotion": sample["emotion2id"]}
        return cls(get_one_fn, emo_id_dict, emo_set_without_neu, wrapped_ds=dataset, **kwargs)


    def __init__(self, get_one_fn, emo_id_dict, emo_set_without_neu, select_n=20000, alpha=0.2, rand_lam_per_batch=True, device='cpu', wrapped_ds=None):
        self.alpha = alpha
        self.rand_lam_per_batch = rand_lam_per_batch
        self.device = device
        self.select_n = select_n
        
        self.get_one_fn = get_one_fn
        self.emo_id_dict = emo_id_dict
        self.emo_set_without_neu = emo_set_without_neu
        self.wrapped_ds = wrapped_ds

    def __getitem__(self, idx):
        # neu_id, non_neu_id = self.pair_ids[idx]
        neu_id = random.choice(self.emo_id_dict["neutral"])
        emo_type = random.choice(list(self.emo_set_without_neu))
        emo_id = random.choice(self.emo_id_dict[emo_type])

        neu = self.get_one_fn(neu_id)
        emo = self.get_one_fn(emo_id)

        neu_mel = torch.from_numpy(neu["mel"])
        emo_mel = torch.from_numpy(emo["mel"])
        neu_pitch = torch.from_numpy(neu["pitch"])
        emo_pitch = torch.from_numpy(emo["pitch"])
        neu_energy = torch.from_numpy(neu["energy"])
        emo_energy = torch.from_numpy(emo["energy"])
        y_neu = neu["emotion"]
        y_emo = emo["emotion"]

        return neu_mel, emo_mel, None, None, y_neu, y_emo, neu_pitch, emo_pitch, neu_energy, emo_energy

    def __len__(self):
        # return len(self.pair_ids)
        return self.select_n

    def collate_fn(self, batch):
        # xi, xj, lam_i, lam_j, y_neu, y_emo = list(zip(*batch))
        neu_mels, emo_mels, _, _, y_neus, y_emos, neu_pitch, emo_pitch, neu_energy, emo_energy = list(zip(*batch))
        batch_size = len(neu_mels)
        lam_i = np.random.beta(self.alpha, self.alpha) if self.alpha != 0 else 0
        lam_j = np.random.beta(self.alpha, self.alpha) if self.alpha != 0 else 0
        if lam_i > 0.5:
            lam_i = 1 - lam_i
        if lam_j < 0.5:
            lam_j = 1 - lam_j
        lam_i = [lam_i] * batch_size
        lam_j = [lam_j] * batch_size

        # print(lam)
        xis = []
        xjs = []
        xi_lens = []
        xj_lens = []
        eis = []
        ejs = []
        pis = []
        pjs = []
        # cut to the same length
        for i in range(len(neu_mels)):
            if self.rand_lam_per_batch == False:
                _lam_i = np.random.beta(self.alpha, self.alpha)
                _lam_j = np.random.beta(self.alpha, self.alpha)
                lam_i[i] = _lam_i
                lam_j[i] = _lam_j
            min_mel_len = min(neu_mels[i].size(0), emo_mels[i].size(0))
            neu_mel = neu_mels[i][:min_mel_len]
            emo_mel = emo_mels[i][:min_mel_len]
            _neu_pitch = neu_pitch[i][:min_mel_len]
            _emo_pitch = emo_pitch[i][:min_mel_len]
            _neu_energy = neu_energy[i][:min_mel_len]
            _emo_energy = emo_energy[i][:min_mel_len]
            xi, _ = mixup_data(emo_mel, neu_mel, lam=lam_i[i])
            xj, _ = mixup_data(emo_mel, neu_mel, lam=lam_j[i])
            pitch_i, _ = mixup_data(_emo_pitch, _neu_pitch, lam=lam_i[i])
            pitch_j, _ = mixup_data(_emo_pitch, _neu_pitch, lam=lam_j[i])
            energy_i, _ = mixup_data(_emo_energy, _neu_energy, lam=lam_i[i])
            energy_j, _ = mixup_data(_emo_energy, _neu_energy, lam=lam_j[i])
            eis.append(energy_i)
            ejs.append(energy_j)
            pis.append(pitch_i)
            pjs.append(pitch_j)
            xis.append(xi)
            xjs.append(xj)
            # xi_lens.append(len(xi))
            # xj_lens.append(len(xj))

        xi_lens = [len(x) for x in xis]
        xj_lens = [len(x) for x in xjs]
        max_xi_len = max(xi_lens)
        max_xj_len = max(xj_lens)
        xi = [F.pad(x, (0, 0, 0, max_xi_len - len(x))) for x in xis]
        xj = [F.pad(x, (0, 0, 0, max_xj_len - len(x))) for x in xjs]
        ei = [F.pad(x, (0, max_xi_len - len(x)), mode="constant") for x in eis]
        ej = [F.pad(x, (0, max_xj_len - len(x)), mode="constant") for x in ejs]
        pi = [F.pad(x, (0, max_xi_len - len(x)), mode="constant") for x in pis]
        pj = [F.pad(x, (0, max_xj_len - len(x)), mode="constant") for x in pjs]
        xi = torch.stack(xi)
        xj = torch.stack(xj)
        ei = torch.stack(ei)
        ej = torch.stack(ej)
        pi = torch.stack(pi)
        pj = torch.stack(pj)
        lam_i = torch.tensor(lam_i, requires_grad=False)
        lam_j = torch.tensor(lam_j, requires_grad=False)
        y_neus = torch.tensor(y_neus, requires_grad=False)
        y_emos = torch.tensor(y_emos, requires_grad=False)
        return xi, xj, pi, pj, ei, ej, lam_i, lam_j, xi_lens, xj_lens, y_neus, y_emos


def get_loaders(configs, device, batch_size):
    preprocess_config, model_config, train_config = configs
    _train_set = EmoFS2Dataset(
        "train.txt", 
        preprocess_config, 
        train_config, sort=False, drop_last=False
    )
    _val_set = EmoFS2Dataset(
        "val.txt", 
        preprocess_config, 
        train_config, 
        sort=False, 
        drop_last=False,
    )
    train_set = MixDataset.from_emofs2_ds(_train_set, select_n=50000, alpha=1, device=device)
    val_set = MixDataset.from_emofs2_ds(_val_set, select_n=300, alpha=1, device=device)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_set.collate_fn,
        num_workers=12,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_set.collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader, train_set, val_set, _train_set, _val_set


def get_es_loaders(configs, device, batch_size):
    ds_dir = "./datasets/esd_processed/mel"
    list_files = get_list_filesss(os.path.join(ds_dir))
    val_spks = ["0011", "0001", "0015", "0005"]
    list_train = list(filter(lambda f: f.replace(".pkl", "").split("_")[1] not in val_spks, list_files))
    list_val = list(filter(lambda f: f.replace(".pkl", "").split("_")[1] in val_spks, list_files))
    train_ds = ESDataset(list_train, base_dir=ds_dir)
    val_ds = ESDataset(list_val, base_dir=ds_dir)
    mix_train_ds = MixDataset.from_es_ds(train_ds, select_n=50000, alpha=1, device=device)
    mix_val_ds = MixDataset.from_es_ds(val_ds, select_n=300, alpha=1, device=device)
    train_loader = torch.utils.data.DataLoader(
        mix_train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mix_train_ds.collate_fn,
        drop_last=True,
        # num_workers=36,
        # num_workers=6,
    )
    val_loader = torch.utils.data.DataLoader(
        mix_val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mix_val_ds.collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader, mix_train_ds, mix_val_ds, train_ds, val_ds
    

if __name__ == "__main__":
    from config import read_config
    configs = read_config()
    train_loader, val_loader, _, _, _, _ = get_es_loaders(configs, device="cpu")
    train_loader.dataset.alpha=0
    for batch in train_loader:
    #     # print(batch)
        xi, xj, pi, pj, ei, ej, lam_i, lam_j, xi_lens, xj_lens, y_neus, y_emos = batch
        print(xi.shape, xj.shape, pi.shape, pj.shape, ei.shape, ej.shape, lam_i, lam_j, xi_lens, xj_lens, y_neus, y_emos)
        break
    #     # emo_count = {}
    #     # for emo in y_emos:
    #     #     emo = emo.item()
    #     #     if emo not in emo_count:
    #     #         emo_count[emo] = 0
    #     #     emo_count[emo] += 1
    #     # print(emo_count)
    #     print(lam_j)
    #     break
    # from sklearn.model_selection import train_test_split
    
    
    # ds_dir = "./esd_dataset_processed"
    # list_files = get_list_filesss(ds_dir)
    # list_train, list_val = train_test_split(list_files, test_size=0.1, random_state=42)
    # print(list_train)
    
    
    # # print(len(list_files), len(list_train), len(list_val))
    # train_ds = ESDataset(list_train, base_dir=ds_dir)
    # val_ds = ESDataset(list_val, base_dir=ds_dir)
    # # print(train_ds.__getitem__(0))
    # mix_train_ds = MixDataset.from_es_ds(train_ds, select_n=50000, alpha=1, device="cpu")
    # print(mix_train_ds.__getitem__(0))