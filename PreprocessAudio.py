import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import math
import numpy as np
import pandas as pd
from pydub import AudioSegment
from sklearn.linear_model import LinearRegression

def conv2wav_torch(audio_dir, resample):
    mp3 = AudioSegment.from_mp3(audio_dir)
    wav = mp3.export(f'{audio_dir[:-4]}.wav', format="wav")
    waves, sr = torchaudio.load(f'{audio_dir[:-4]}.wav')
    waves = F.resample(waves, orig_freq=sr, new_freq=resample)
    return waves, resample

def vad_torch(waves, buff_size, threshold, display_info=False):
    mono_signal = waves[0].numpy()
    total_sig   = int(mono_signal.shape[0]/buff_size)
    signal      = np.array([])
    for i in range(total_sig):
        sig = mono_signal[i*buff_size:(i+1)*buff_size]
        rms = math.sqrt(np.square(sig).mean())
        if(rms > threshold):
            signal = np.append(signal,sig)
    signal = signal.astype('float')
    if (display_info):
        print("Number of total signal (signal_arr/buff_size):", total_sig)
        print("Signal data type:", signal.dtype)
        print(f"Signal shape: ({signal.shape})")
    return torch.tensor([signal])

def calculate_distance(x, y, m, c):
    return abs(y - (m * x + c)) / np.sqrt(m**2 + 1)

def remove_outliers(df, P):
    m, c = np.polyfit(df['len_transkrip'], df['len_mfcc'], 1)
    df['distance'] = calculate_distance(df['len_transkrip'], df['len_mfcc'], m, c)
    return df[df['distance'] <= P].drop(columns=['distance'])

def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0
    
    while low <= high:
        mid = (high + low) // 2
        if arr[mid] < x:
            low = mid + 1
 
        elif arr[mid] > x:
            high = mid - 1
 
        else:
            return mid

    return -1

class PreprocessAudio:
    def __init__(self, audio_dir, transcript_df, P, n_fft = 1024,
                 win_length = None, hop_length = 128, n_mels = 64, n_mfcc = 64):
        self.audio_dir = audio_dir
        self.transcript_df = transcript_df
        self.P = P
        self.mfcc_transform = T.MFCC(sample_rate=8000,
                                     n_mfcc=n_mfcc,
                                     melkwargs={"n_fft": n_fft,
                                                "n_mels": n_mels,
                                                "hop_length": hop_length,
                                                "mel_scale": "htk",
                                               })
    def load_audio(self):
        i = 0
        print("Mounted audio directory at:", self.audio_dir)
        transcript_df = self.transcript_df
        _dir = self.transcript_df['path']
        dataset = []
        list_file_error = []
        list_index_error = []
        for mp3 in _dir:
            try:
                if (mp3[-3:] == 'wav'):
                    continue
                mp3 = self.audio_dir + mp3
                waves, sr = conv2wav_torch(mp3, 8000)
                signal_vad = vad_torch(waves, 1000, 0.012)
                mfcc = self.mfcc_transform(signal_vad.type(torch.float))
                mfcc = mfcc.permute(0, 2, 1)
                # print(mfcc.shape)
                dataset.append(mfcc.numpy().tolist())
                os.unlink(mp3)
                i += 1
            except:
                print(f"Error di file {mp3}")
                print(f"Counter di {i}")
                list_file_error.append(mp3)
                list_index_error.append(i)
                transcript_df = transcript_df.drop(i)
                i += 1
                continue
            
        transcript_df = transcript_df.reset_index(drop=True)
        
        list_len_data = []
        for i in range(len(dataset)):
            list_len_data.append(np.array(dataset[i][0]).shape[0])

        list_len_transkrip = []
        for text in transcript_df['sentence']:
            list_len_transkrip.append(len(text))

        df_komparasi_len = pd.DataFrame({'len_mfcc':list_len_data, 'len_transkrip':list_len_transkrip})
        
        linreg_model = LinearRegression()
        linreg_model.fit(np.array(df_komparasi_len['len_transkrip']).reshape(-1, 1), np.array(df_komparasi_len['len_mfcc']).reshape(-1,1))
        preds = linreg_model.predict(np.array(df_komparasi_len['len_transkrip']).reshape(-1, 1))

        df_komp_copy = df_komparasi_len.copy()
        df_komp_cleaned = remove_outliers(df_komp_copy, self.P)
        
        cleaned_index = pd.Series(df_komp_cleaned.index.tolist())
        uncleaned_index = pd.Series(transcript_df.index.tolist())
        not_cleaned_index = uncleaned_index[~uncleaned_index.isin(cleaned_index)].tolist()
        
        df_filtered_cleaned = transcript_df.loc[cleaned_index].reset_index(drop=True)
        
        dataset_cleaned = []
        for i in range(len(dataset)):
            pos = binary_search(not_cleaned_index, i)
            if pos != -1:
                continue
            
            dataset_cleaned.append(dataset[i])

        del dataset
        
        return dataset_cleaned, df_filtered_cleaned