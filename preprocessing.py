import os
import re
import csv
import shutil
import random
import multiprocessing
from functools import partial
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from utils import load_audio, STR_CH_FIRST, SPEECH_SAMPLE_RATE

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def speech_resampler(fname, dataset_path, _id):
    src, _ = load_audio(
        path=fname,
        ch_format= STR_CH_FIRST,
        sample_rate= SPEECH_SAMPLE_RATE,
        downmix_to_mono= True
        )
    save_name = os.path.join(dataset_path, "feature/npy/" , f"{_id}.npy")
    np.save(save_name, src.astype(np.float32))
    return src

def valid_datasplit(df_train):
    label_list = []
    for idx in range(len(df_train)):
        item = df_train.iloc[idx]
        label_list.append(item.idxmax())
    x_TR, x_VA, _, _ = train_test_split(df_train.index, label_list, stratify=label_list, test_size=0.1, random_state=42)
    return df_train.loc[x_TR], df_train.loc[x_VA]

def HIKIA_processor(dataset_path):
    os.makedirs(os.path.join(dataset_path, "split"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "feature", "npy"), exist_ok=True)
    fnames = [i for i in os.listdir(os.path.join(dataset_path, "wav")) if ".wav" in i]
    items = []
    for fname in tqdm(fnames):
        y = speech_resampler(os.path.join(dataset_path, "wav" ,fname), dataset_path, fname.replace(".wav",""))
        duration = y.shape[-1] / SPEECH_SAMPLE_RATE
        fname = fname.replace(".wav","")
        gen_pid, scene, trial, emo = fname.split("_")
        gender, pid = gen_pid[0], gen_pid[1]
        items.append({
            "fname":fname,
            "gen_pid":gen_pid,
            "gender":gender,
            "pid":pid,
            "scene":scene,
            "trial":trial,
            "emo":emo,
            "duration": duration,
        })
    
    df_items = pd.DataFrame(items).set_index("fname")
    emotino_dict = {
        'a':"angry",  
        'd':"desperate", 
        'e':"excitement", 
        'n':"neutral"
    }
    df_items = df_items.replace({"emo": emotino_dict})
    # df_items.to_csv(os.path.join(dataset_path, "split", "annotation.csv"))
    # lb = preprocessing.LabelBinarizer()
    # binary = lb.fit_transform(df_items['emo'])
    # df_binary = pd.DataFrame(binary, index=df_items.index, columns=lb.classes_)
    # for gen_pid in set(df_items['gen_pid']):
    #     EVAL_item = df_items[df_items['gen_pid'] == gen_pid].index
    #     TRAIN_item = df_items[df_items['gen_pid'] != gen_pid].index
    #     df_test = df_binary.loc[EVAL_item]
    #     df_train, df_valid = valid_datasplit(df_binary.loc[TRAIN_item])
    #     df_train.to_csv(os.path.join(dataset_path, "split", f"{gen_pid}_train.csv"))
    #     df_valid.to_csv(os.path.join(dataset_path, "split", f"{gen_pid}_valid.csv"))
    #     df_test.to_csv(os.path.join(dataset_path, "split", f"{gen_pid}_eval.csv"))
    #     print(df_train.sum(), df_valid.sum(), df_test.sum())
    #     print("="*10)

# def TESS_processor(tess_path):
#     wav_path = os.path.join(tess_path, "wav")
#     tess_directory_list = os.listdir(wav_path)
#     file_emotion = []
#     file_path = []
#     for dirs in tess_directory_list:
#         directories = os.listdir(os.path.join(wav_path,dirs))
#         for file in directories:
#             part = file.split('.')[0]
#             part = part.split('_')[2]
#             if part=='ps':
#                 file_emotion.append('surprise')
#             else:
#                 file_emotion.append(part)
#             file_path.append(os.path.join(dirs, file.replace(".wav","")))
#     # dataframe for emotion of files
#     emotion_df = pd.DataFrame(file_emotion,index=file_path, columns=['Emotions'])
#     y = emotion_df['Emotions']
#     X = emotion_df.index
#     lb = preprocessing.LabelBinarizer()
#     binary = lb.fit_transform(emotion_df)
#     df_binary = pd.DataFrame(binary, index=emotion_df.index, columns = lb.classes_)
#     X_train, X_test, _, _ = train_test_split(X, y, stratify=y ,test_size=0.33, random_state=42)
#     df_binary.loc[X_train].to_csv("./dataset/TESS/split/train.csv")
#     df_binary.loc[X_test].to_csv("./dataset/TESS/split/eval.csv")

#     # items = []
#     # durations = []
#     # for fname in tqdm(emotion_df.index):
#     #     folder = fname.split("/")[0]
#     #     os.makedirs(os.path.join(tess_path, "feature/npy/" ,folder), exist_ok=True)
#     #     y = speech_resampler(os.path.join(wav_path + ".mp3" ,fname), tess_path, fname)
#     #     durations.append(y.shape[-1] / SPEECH_SAMPLE_RATE)
#     # print(np.mean(durations))

def main():
    HIKIA_processor("./dataset")
    # TESS_processor("./dataset/TESS")


if __name__ == '__main__':
    main()