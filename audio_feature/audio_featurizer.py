

import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew , kurtosis



data=[]

def statistics(list, feature, columns_name, data):
    i = 0
    for ele in list:
        _skew = skew(ele)
        columns_name.append(f'{feature}_kew_{i}')
        min = np.min(ele)
        columns_name.append(f'{feature}_min_{i}')
        max = np.max(ele)
        columns_name.append(f'{feature}_max_{i}')
        std = np.std(ele)
        columns_name.append(f'{feature}_std_{i}')
        mean = np.mean(ele)
        columns_name.append(f'{feature}_mean_{i}')
        median = np.median(ele)
        columns_name.append(f'{feature}_median_{i}')
        _kurtosis = kurtosis(ele)
        columns_name.append(f'{feature}_kurtosis_{i}')

        i += 1
        data.append(_skew)
        data.append(min)
        data.append(max)
        data.append(std)
        data.append(mean)
        data.append(median)
        data.append(_kurtosis)
    return data


def extract_features(audio_path, title):

  
    columns_name = ['title']
    data.append(title)

    x , sr = librosa.load(audio_path)

    chroma_stft = librosa.feature.chroma_stft(x, sr)
    stft = statistics(chroma_stft, 'chroma_stft', columns_name, data)
    

    chroma_cqt = librosa.feature.chroma_cqt(x, sr)
    cqt = statistics(chroma_cqt, 'chroma_cqt', columns_name, data)
    

    chroma_cens = librosa.feature.chroma_cens(x, sr)
    cens = statistics(chroma_cens, 'chroma_cens', columns_name, data)
    

    mfcc = librosa.feature.mfcc(x, sr)
    mf = statistics(mfcc, 'mfcc', columns_name, data)
    

    rms = librosa.feature.rms(x, sr)
    rm = statistics(rms, 'rms', columns_name, data)
    

    spectral_centroid = librosa.feature.spectral_centroid(x, sr)
    centroid = statistics(spectral_centroid, 'spectral_centroid', columns_name, data)
    

    spectral_bandwidth = librosa.feature.spectral_bandwidth(x, sr)
    bandwidth = statistics(spectral_bandwidth, 'spectral_bandwidth', columns_name, data)
    

    spectral_contrast = librosa.feature.spectral_contrast(x, sr)
    contrast = statistics(spectral_contrast, 'spectral_contrast', columns_name, data)
    

    spectral_rolloff = librosa.feature.spectral_rolloff(x, sr)
    rolloff = statistics(spectral_rolloff, 'spectral_rolloff', columns_name, data)
    

    tonnetz = librosa.feature.tonnetz(x, sr)
    tonnetz = statistics(tonnetz, 'tonnetz', columns_name, data)
    

    zero_crossing_rate = librosa.feature.zero_crossing_rate(x, sr)
    zero = statistics(zero_crossing_rate, 'zero_crossing_rate', columns_name, data)
   

    return data , columns_name

def final(data,columns_name):

    # combining list of row values (data) and list of columns (columns_name)
    nnn = []
    for i in range(0, len(data), len(columns_name)):
        nnn.append(data[i:i + 519])

    # creating dataframe
    df2 = pd.DataFrame(nnn, columns=columns_name)
    df2 = df2.drop(["title"],axis=1)
    
    return df2 