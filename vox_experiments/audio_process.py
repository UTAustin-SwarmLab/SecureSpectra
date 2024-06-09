import librosa
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
# import signature_module as sm

def split_audio_into_samples(audio_file, sample_duration=5, sr=24000):
    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=sr)
    
    # Calculate the number of samples per segment
    samples_per_segment = sample_duration * sr
    
    # Calculate the total number of segments
    num_segments = int(np.ceil(len(audio) / samples_per_segment))
    
    # Initialize an empty list to store the audio segments
    audio_segments = []
    
    # Split the audio into segments
    for i in range(num_segments):
        # Calculate start and end indices for the current segment
        start = i * samples_per_segment
        end = min(start + samples_per_segment, len(audio))
        
        # Extract the current segment
        segment = audio[start:end]
        
        # If the segment is shorter than the desired duration, pad it with zeros
        if len(segment) < samples_per_segment:
            segment = np.pad(segment, (0, samples_per_segment - len(segment)), mode='constant')
        
        # Append the segment to the list
        audio_segments.append(segment)
    
    return audio_segments



def get_five_sec(audio, sample_duration=5, sr=24000):
    # Load the audio file
    # Calculate the number of samples per segment
    samples_per_segment = round(sample_duration * sr)
    start = 0
    end = min(start + samples_per_segment, len(audio))
    
    # Extract the current segment
    segment = audio[start:end]
    
    # If the segment is shorter than the desired duration, pad it with zeros
    if len(segment) < samples_per_segment:
        segment = np.pad(segment, (0, samples_per_segment - len(segment)), mode='constant')
    

    return segment

# Step 1: Convert audio file to mel spectrogram
def get_spectogram(audio, sr=24000, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_db = np.array(mel_spectrogram_db)
    return mel_spectrogram_db


def get_database(audio_dir,  target_value, save_loc,sec=5.44, sr=24000,n_mels=128):
    
    mel_specs = []
    target_values = []
    for audio_path in tqdm(audio_dir):
        audio, sr = librosa.load(audio_path, sr=sr)
        audio = get_five_sec(audio,sec)
        mel_specs.append(get_spectogram(audio,sr,n_mels))
        target_values.append(target_value) # for original audio
    mel_specs = np.array(mel_specs)
    target_values = np.array(target_values)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    np.save(save_loc+"/melspecs.npy",mel_specs)
    np.save(save_loc+"/labels.npy", target_values)
    print("Dataset is created.")
    
def save_spectrogram_image(spectrogram, output_file, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_file)  # Save the image to a file
    plt.close()  # Close the plot to free up memory

    # # Example usage
    # spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    # save_spectrogram_image(spectrogram, 'spectrogram.png')

# def get_signed_database(audio_dir, save_loc, target_value=1, sr=24000,n_mels=128):
#     # variable initialization
#     mel_specs = []
#     target_values = []
#     # signature module
#     for file in audio_dir:
#         audio, sr = librosa.load(file, sr=sr)
#         mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
#         mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#         mel_spec_db = np.array(mel_spec_db)
#         mel_specs.append(sm.signed_melspecs(audio))
#         target_values.append(1)
#     mel_specs = np.array(mel_specs)
#     target_values = np.array(target_values)
#     if not os.path.exists(save_loc):
#         os.makedirs(save_loc)
#     np.save(save_loc+"/melspecs.npy",mel_specs)
#     np.save(save_loc+"/labels.npy", target_values)
#     print("Dataset is created.")
        
    
    
    
    
