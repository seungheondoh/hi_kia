import os
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor


class Emotion_Dataset(Dataset):
    def __init__(self, root, split, data_type, cv_split):
        self.root = root
        self.split = split
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        self.maxseqlen = 16000 * 16
        self.data_type = data_type
        self.cv_split = cv_split # "01F"
        # self.df_meta = pd.read_csv(os.path.join(self.root, f"split/annotation.csv"), index_col=0)
        if split == 'TRAIN':
            self.fl = pd.read_csv(os.path.join(self.root, f"split/{self.cv_split}_train.csv"), index_col=0)
        elif split == 'VALID':
            self.fl = pd.read_csv(os.path.join(self.root, f"split/{self.cv_split}_valid.csv"), index_col=0)
        elif split == 'TEST':
            self.fl = pd.read_csv(os.path.join(self.root, f"split/{self.cv_split}_eval.csv"), index_col=0)
            
    def __getitem__(self, indices):
        audios, labels, fnames, binarys = [], [], [], []
        for index in indices:
            item = self.fl.iloc[index]
            fname = item.name
            binary = item.values
            audio = np.load(os.path.join(self.root, f"feature/npy/{fname}.npy"))
            audios.append(audio.squeeze(0)) # delete channel 
            labels.append(torch.from_numpy(binary))
            fnames.append(fname) 
        audio_encoding = self.processor(audios, return_tensors='pt',sampling_rate=16000, padding=True, return_attention_mask=True)
        audios = audio_encoding['input_values']
        audio_mask = audio_encoding['attention_mask']
        # audios = self.processor_fn(audios)
        labels = torch.stack(labels, dim=0).float()
        return {"audios" : audios,"audio_mask": audio_mask, "labels" : labels, "fnames": fnames}
    
    def processor_fn(self, batch):
        # if you use processor, don't need pad_fn
        longest = max([len(x) for x in batch])
        target_seqlen = min(self.maxseqlen, longest)
        batch_stack = []
        for x in batch:
            if len(x) > target_seqlen:
                time_ix = int(np.floor(np.random.random(1) * (len(x) - target_seqlen)))
                batch_stack.append(x[time_ix:time_ix+target_seqlen])
            else:
                batch_stack.append(np.pad(x, (0, target_seqlen - len(x))))
        return torch.from_numpy(np.stack(batch_stack))

    def __len__(self):
        return len(self.fl)



# class Emotion_Dataset(Dataset):
#     def __init__(self, root, split, data_type, cv_split):
#         self.root = root
#         self.split = split
#         self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
#         self.maxseqlen = 16000 * 16
#         self.data_type = data_type
#         self.cv_split = cv_split # "01F"
#         # self.df_meta = pd.read_csv(os.path.join(self.root, f"split/annotation.csv"), index_col=0)
#         if split == 'TRAIN':
#             self.fl = pd.read_csv(os.path.join(self.root, "TESS/split/train.csv"), index_col=0)
#         elif split == 'VALID':
#             self.fl = pd.read_csv(os.path.join(self.root, "TESS/split/eval.csv"), index_col=0)
#         elif split == 'TEST':
#             self.fl = pd.read_csv(os.path.join(self.root, f"split/{self.cv_split}_eval.csv"), index_col=0)
            
#     def __getitem__(self, indices):
#         audios, labels, fnames, binarys = [], [], [], []
#         for index in indices:
#             item = self.fl.iloc[index]
#             fname = item.name
#             binary = item.values
#             audio = np.load(os.path.join(self.root, f"TESS/feature/npy/{fname}.npy"), mmap_mode='r')
#             audios.append(audio.squeeze(0)) # delete channel 
#             labels.append(torch.from_numpy(binary))
#             fnames.append(fname) 
#         audios = self.processor_fn(audios)
#         labels = torch.stack(labels, dim=0).float()
#         return {"audios" : audios,"labels" : labels, "fnames": fnames}
    
#     def processor_fn(self, batch):
#         # if you use processor, don't need pad_fn
#         longest = max([len(x) for x in batch])
#         target_seqlen = min(self.maxseqlen, longest)
#         batch_stack = []
#         for x in batch:
#             if len(x) > target_seqlen:
#                 time_ix = int(np.floor(np.random.random(1) * (len(x) - target_seqlen)))
#                 batch_stack.append(x[time_ix:time_ix+target_seqlen])
#             else:
#                 batch_stack.append(np.pad(x, (0, target_seqlen - len(x))))
#         return torch.from_numpy(np.stack(batch_stack))

#     def __len__(self):
#         return len(self.fl)