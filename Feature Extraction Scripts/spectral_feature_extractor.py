import sys
import csv
import numpy as np
import librosa
import librosa.display

# Name of the csv file goes here
csv_name = 'C:/Language_Audio_Samples/english_male/english_male_voice_data.csv'

# Opening the csv_file
csv_file = open(csv_name, 'w', newline='')

# csv header line containing the feature names
header = 'MFCC[0] '
for i in range(1,39):
    header += f'MFCC[{i}] '
header +=  'Spectral_Bandwidth Spectral_Centroid Spectral_Contrast Spectral_Flatness Spectral_Rolloff Tempogram Zero_Crossing_Rate Label'

# Writing the header line to the csv file
csv.writer(csv_file).writerow(header.split())

print('Extracting Features and Writing to File: ' + csv_name)

# For each 10 second clip:
for n in range(0,120):
    # Open the audio clip
    audio_clip = f'C:/Language_Audio_Samples/english_male/{n}_english_male.wav'
    x, sr = librosa.load(audio_clip, sr = None) 
    
    # Extract features using librosa
    mfcc            = librosa.feature.mfcc(x, n_mfcc = 39)
    spect_bandwidth = librosa.feature.spectral_bandwidth(x)
    spect_centroid  = librosa.feature.spectral_centroid(x)
    spect_contrast  = librosa.feature.spectral_contrast(x)
    spect_flatness  = librosa.feature.spectral_flatness(x)
    spect_rolloff   = librosa.feature.spectral_rolloff(x)
    tempo           = librosa.feature.tempogram(x)
    zero_cross_rate = librosa.feature.zero_crossing_rate(x)
    
    # Append the mean of each feature to the csv_file
    csv_file = open(csv_name, 'a', newline='')
    with csv_file:
        features = f'{np.mean(mfcc[0])} '
        for i in range(1,39):
            features += f'{np.mean(mfcc[i])} '
        features += f'{np.mean(spect_bandwidth)} {np.mean(spect_centroid)} {np.mean(spect_contrast)}  {np.mean(spect_flatness)} {np.mean(spect_rolloff)} {np.mean(tempo)} {np.mean(zero_cross_rate)} '
        csv.writer(csv_file).writerow(features.split())

print("Feature Extraction Complete")