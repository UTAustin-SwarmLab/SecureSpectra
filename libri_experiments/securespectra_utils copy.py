from models import SignatureNet
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
from random import randint
from random import seed
from models import VerifierNet, SignatureNet
import json
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

def load_and_infer(input_array, model_path="./best_model_sign_1.pt", device='cuda:0'):
    """
    Load a PyTorch model from a .pth file and perform inference on a numpy array.
    
    Args:
        model_path (str): Path to the .pth file containing the model state_dict.
        input_array (numpy.ndarray): Input data for inference.
        device (str): Device to perform inference on ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: Output tensor from the model's inference.
    """
    # Load the model
    model = SignatureNet()  # Replace YourModel with your model class
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Convert input array to torch tensor
    input_tensor = torch.tensor(input_array).unsqueeze(0).unsqueeze(0)
    print("input shape: "+str(input_tensor.shape))
    input_tensor = input_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    return output_tensor


def mfcc_normalize(mfcc):
    mfcc= (mfcc  - np.min(mfcc )) / (np.max(mfcc ) - np.min(mfcc))
    return mfcc


def mfcc_denormalize(input_array, min_value, max_value):
    """
    Scale the input array to have a specified minimum and maximum range.
    
    Args:
        input_array (numpy.ndarray): Input data to be scaled.
        min_value (float): Desired minimum value after scaling.
        max_value (float): Desired maximum value after scaling.
    
    Returns:
        numpy.ndarray: Scaled input array.
    """
    min_input = input_array.min()
    max_input = input_array.max()
    scaled_array = min_value + (input_array - min_input) * (max_value - min_value) / (max_input - min_input)
    return scaled_array

def calculate_accuracy(predictions, labels):
    """
    Calculate the accuracy of the predictions.
    Args:
        predictions (torch.Tensor): Predicted labels.
        labels (torch.Tensor): Ground truth labels.
    Returns:
        float: Accuracy of the predictions.
    """
    # Calculate accuracy on GPU
    accuracy = ((predictions>0.5).int() == labels).float().mean()

    return accuracy.item()

def save_history(verificationLossVal,signatureLossVal,verificationLossTrain,signatureLossTrain,verificationAccuracyVal,target_path="history.json"):
    # Example dictionary of NumPy arrays
    data_dict = {
        'verificationLossVal': verificationLossVal,
        'signatureLossVal': signatureLossVal,
        'verificationLossTrain': verificationLossTrain,
        'signatureLossTrain':signatureLossTrain,
        'verificationAccuracyVal':verificationAccuracyVal
    }

    # Define the path to save the JSON file
    json_file_path =target_path
    # Write the dictionary into a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data_dict, json_file)
    print("Dictionary of histories is saved successfully...: "+str(target_path))
    

class signatureDataset(Dataset):
    def __init__(self, mfcc_file):
        self.mfcc = np.load(mfcc_file)
        self.mfcc = (self.mfcc  - np.min(self.mfcc )) / (np.max(self.mfcc ) - np.min(self.mfcc)) # normalization of mel specs for better learning
        
    def __len__(self):
        return self.mfcc.shape[0]

    def __getitem__(self, idx):
        mfcc_sample = torch.tensor(self.mfcc[idx]).float().unsqueeze(0)
        return mfcc_sample 

def initialize_weights_xavier(m):
    if isinstance(m, nn.Linear):
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(m.weight)
        # Initialize biases to zero
        nn.init.constant_(m.bias, 0)
        
def dataEvaluateEER(verifyModelDir, melData, labels, device, isPrint=True):
    # load the model
    verifyModel = VerifierNet()
    verifyModel.load_state_dict(torch.load(verifyModelDir))
    verifyModel.to(device)
    verifyModel.eval()
    # create a data loader
    val_dataset = evalDataset(melData, labels)
    valid_loader = DataLoader(val_dataset, batch_size=256)
    predictions_all = np.array([])
    labels_all = labels 
    # perform inference 
    for i_v, (audios_v, labels_v) in tqdm(enumerate(valid_loader)):
        # print(i_v)
        # validation only for the signature model
        audios_v = audios_v.to(device)
        predictions = verifyModel(audios_v)
        predictions_all = np.append(predictions_all, predictions.detach().cpu().numpy().flatten())


    fpr, tpr, _ = roc_curve(labels_all, predictions_all, pos_label=1)
    spf_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    
    
    
    if isPrint:
        print(f"[Verification EER: {spf_eer}]")
        print(f"[Verification Accuracy]: {((predictions_all>0.5).astype(int) == labels_all).mean()}]")
    return spf_eer

def dataEvaluate(verifyModelDir, melData, labels, device, isPrint=True):
    # load the model
    verifyModel = VerifierNet()
    verifyModel.load_state_dict(torch.load(verifyModelDir))
    verifyModel.to(device)
    verifyModel.eval()
    # create a data loader
    val_dataset = evalDataset(melData, labels)
    valid_loader = DataLoader(val_dataset, batch_size=256)
    epochValAccVerify=0
    # perform inference 
    for i_v, (audios_v, labels_v) in enumerate(valid_loader):
        # print(i_v)
        # validation only for the signature model
        audios_v = audios_v.to(device)
        labels_v = labels_v.to(device)
        predictions = verifyModel(audios_v)

        epochValAccVerify+=calculate_accuracy(predictions, labels_v)
    
    # get the result
    epochValAccVerify=epochValAccVerify/(i_v+1)
    if isPrint:
        print(f"[Verification Accuracy: {epochValAccVerify}]")
    return epochValAccVerify

class evalDataset(Dataset):
    def __init__(self, mfcc_file, labels):
        self.mfcc = mfcc_file
        self.labels = labels
        self.mfcc = (self.mfcc  - np.min(self.mfcc )) / (np.max(self.mfcc ) - np.min(self.mfcc)) # normalization of mel specs for better learning
            
    def __len__(self):
        return self.mfcc.shape[0]

    def __getitem__(self, idx):
        mfcc_sample = torch.tensor(self.mfcc[idx]).float().unsqueeze(0)
        labels_sample = torch.tensor(self.labels[idx]).float()
        return mfcc_sample, labels_sample

def signData(signModelDir, melDataDir, device):
    # load the model
    signModel = SignatureNet()
    signModel.load_state_dict(torch.load(signModelDir))
    signModel.to(device)
    signModel.eval()
    # create a data loader
    melDataset = signatureDataset(melDataDir)
    sign_loader = DataLoader(melDataset,batch_size=256)
    signedAudios = np.empty((0,)) 
    for i_v, (audios_v) in tqdm(enumerate(sign_loader)):
        audios_v = audios_v.to(device)
        signedAudio = signModel(audios_v)
        signedAudio_np = signedAudio.detach().cpu().numpy() 
        if i_v == 0:
            # Initialize the shape after the first model output is known
            signedAudios = np.empty((0,) + signedAudio_np.shape[1:])
        signedAudios = np.append(signedAudios, signedAudio_np, axis=0)
    return signedAudios


def add_dp_noise(binary_vector, epsilon, global_sensitivity):
    # Ensure the binary_vector is of the correct size
    if binary_vector.size != 32:
        raise ValueError("binary_vector must be of size 32")
    
    # Calculate the scale parameter for the Laplace distribution
    b = global_sensitivity / epsilon
    
    # Generate noise from the Laplace distribution
    noise = np.random.laplace(0, b, size=binary_vector.shape)
    
    # Add noise to the binary vector
    noisy_vector = binary_vector + noise
    
    return noisy_vector

# # Example usage:
# binary_vector = np.random.choice([0, 1], size=32)
# epsilon = 30  # Adjust epsilon as needed for your privacy-accuracy trade-off
# noisy_vector = add_dp_noise(binary_vector, epsilon, 0.25)
# print("Original Vector:", binary_vector)
# print("Noisy Vector:", noisy_vector)