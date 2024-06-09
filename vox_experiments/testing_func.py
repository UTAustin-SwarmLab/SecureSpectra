from securespectra_utils import *
import numpy as np
import os
from tqdm import tqdm


def verificationOnlyTesting(verifyModelDir = "./best_model_verify_only.pt", originalMelDataDir = "../models/original_CV_val/melspecs.npy", clonesMelDataDir= "/home/obiwan/repos/watermarking_module/cloned_audio/cloned_CV_val/melspecs.npy"):
    # set the directory to the right place
    
    
    
    originalMelData = normalizeMel(np.load(originalMelDataDir))
    originalMelLabels = np.ones(originalMelData.shape[0])
   
    clonesMelData = normalizeMel(np.load(clonesMelDataDir))
    clonesMelLabels = np.zeros(clonesMelData.shape[0])
        
    evalX = np.concatenate((originalMelData, clonesMelData), axis=0)
    evalY = np.concatenate((originalMelLabels, clonesMelLabels), axis=0)
    EER=dataEvaluateEER(verifyModelDir=verifyModelDir, melData=evalX, labels=evalY, device=torch.device("cuda:0"), isPrint=True)

def normalizeMel(mfcc):
    return (mfcc  - np.min(mfcc )) / (np.max(mfcc ) - np.min(mfcc))

def general_testing_func_EER(signModelDir = "./best_model_sign_vox.pt", verifyModelDir = "./best_model_verify_vox.pt",mel_data_dir="./training_mels_vox/trainingMels.npy"):
    # set the directory to the right place

    originalMelData = np.load(mel_data_dir)
    originalMelLabels = np.zeros(originalMelData.shape[0])
    num_samples = int(originalMelLabels.shape[0]*0.9)
    originalMelData = originalMelData[num_samples:]
    originalMelLabels = originalMelLabels[num_samples:]
    # code for the audio signing

    
    
    signedAudios = signData(signModelDir=signModelDir,melDataDir=mel_data_dir, device=torch.device("cuda:0"))
    signedAudios = signedAudios.squeeze(1)
    print(signedAudios.shape)
    np.save("signedMels.npy",signedAudios)

    signedMelDataDir = "signedMels.npy"
    signedMelData = np.load(signedMelDataDir)
    signedMelLabels = np.ones(signedMelData.shape[0])
    
    
    originalMelData = normalizeMel(originalMelData)
    signedMelData = normalizeMel(signedMelData)

    
    evalX = np.concatenate((originalMelData, signedMelData), axis=0)
    evalY = np.concatenate((originalMelLabels, signedMelLabels), axis=0)
    
    

    dataEvaluateEER(verifyModelDir=verifyModelDir, melData=evalX, labels=evalY, device=torch.device("cuda:0"), isPrint=True)
    
    

    
def general_testing_func():
    # set the directory to the right place
    os.chdir("/home/obiwan/repos/watermarking_module/")

    # code for the audio signing
    melDataDir = "/home/obiwan/repos/models/original_CV_val/melspecs.npy"
    signModelDir = "pretrained_models/signature_model.pt"
    # modelName = 9
    signedAudios = signData(signModelDir=signModelDir,melDataDir=melDataDir, device=torch.device("cuda:0"))
    signedAudios = signedAudios.squeeze(1)
    print(signedAudios.shape)
    np.save("signedMels.npy",signedAudios)
    # np.save("signedMels_model_"+str(modelName)+"_.npy",signedAudios)


    # code for the verification
    melDataDir = "../models/original_CV_val/melspecs.npy"
    verifyModelDir = "./pretrained_models/verification_model.pt"

    melData = np.load(melDataDir)
    labels = np.zeros(melData.shape[0])

    accuracy=dataEvaluate(verifyModelDir, melData, labels, torch.device("cuda:0"), isPrint=True)

    melDataDir = "signedMels.npy"
    melData = np.load(melDataDir)
    labels = np.ones(melData.shape[0])
    accuracy=dataEvaluate(verifyModelDir, melData, labels, torch.device("cuda:0"), isPrint=True)
    # all functions tested

    melDataDir = "/home/obiwan/repos/watermarking_module/cloned_audio/cloned_CV_val/melspecs.npy"
    melData = np.load(melDataDir)
    labels = np.zeros(melData.shape[0])
    accuracy=dataEvaluate(verifyModelDir, melData, labels, torch.device("cuda:0"), isPrint=True)

def sign_individual_users():
    # set the directory to the right place
    mels_folder = "/home/obiwan/repos/watermarking_module/mel_spectograms/original_clients"
    clients = os.listdir(mels_folder)
    signModelDir = "pretrained_models/signature_model.pt"
    target_folder = "/home/obiwan/repos/watermarking_module/mel_spectograms/signed_clients"
    # code for the audio signing
    for client in tqdm(clients):
        melDataDir = mels_folder+"/"+client
        signedAudios = signData(signModelDir=signModelDir,melDataDir=melDataDir, device=torch.device("cuda:0"))
        signedAudios = signedAudios.squeeze(1)
        # print(signedAudios.shape)
        np.save(target_folder+"/"+client,signedAudios)

#set the directory to the right place
# os.chdir("/home/obiwan/repos/watermarking_module/")        
# verifyModelDir = "./pretrained_models/verification_model.pt"
# mels_folder = "/home/obiwan/repos/watermarking_module/mel_spectograms/original_clients"
# mels_folder_signed = "/home/obiwan/repos/watermarking_module/mel_spectograms/signed_clients"
# mels_folder_cloned = "/home/obiwan/repos/watermarking_module/mel_spectograms/cloned_clients"
# clients = os.listdir(mels_folder)
# accuracies = []
# # verify unsigned
# for client in tqdm(clients):
#     if client!="ae870c12258648e14a66c1e7c5c210b4f1a64b8d01cd62e61e377581c4d427f2b6fd0478eb283bb57692c5dd76a85338b0e8b2aadf47da254d6d2d5d49902007_mel_spectrograms.npy":
#         melDataDir = mels_folder+"/"+client
#         melDataDirSigned = mels_folder_signed+"/"+client
#         melDataDirCloned = mels_folder_cloned+"/"+client
#         melData = np.load(melDataDir)
#         melDataSigned = np.load(melDataDirSigned)
#         melDataCloned = np.load(melDataDirCloned)
#         labels = np.zeros(melData.shape[0])
#         labels_signed = np.ones(melDataSigned.shape[0])
#         labels_cloned = np.zeros(melDataCloned.shape[0])
#         accuracy=dataEvaluate(verifyModelDir, melData, labels, torch.device("cuda:0"), isPrint=False)
#         accuracySigned = dataEvaluate(verifyModelDir, melDataSigned, labels_signed, torch.device("cuda:0"), isPrint=False)
#         accuracyCloned = dataEvaluate(verifyModelDir, melDataCloned, labels_cloned, torch.device("cuda:0"), isPrint=False)
#         accuracy = (accuracyCloned + accuracySigned + accuracy)/3
#         accuracies.append(accuracy)
# print(accuracies)
# print(np.mean(np.array(accuracies)))
# np.save("CommonVoice_SS.npy",np.array(accuracies))

# general_testing_func_EER(signModelDir = "pretrained_models/signature_model.pt", verifyModelDir = "./pretrained_models/verification_model.pt", originalMelDataDir = "../models/original_CV_val/melspecs.npy", clonesMelDataDir= "/home/obiwan/repos/watermarking_module/cloned_audio/cloned_CV_val/melspecs.npy")

# os.chdir("/home/obiwan/repos/watermarking_module/")        
# verifyModelDir = "./pretrained_models/verification_model.pt"
# mels_folder = "/home/obiwan/repos/watermarking_module/mel_spectograms/original_clients"
# mels_folder_signed = "/home/obiwan/repos/watermarking_module/mel_spectograms/signed_clients"
# mels_folder_cloned = "/home/obiwan/repos/watermarking_module/mel_spectograms/cloned_clients"
# clients = os.listdir(mels_folder)
# accuracies = []
# # verify unsigned
# evalX = np.empty((1,1,1))
# evalY = np.array([])
# isFirst = True
# for client in tqdm(clients):
#     if client!="ae870c12258648e14a66c1e7c5c210b4f1a64b8d01cd62e61e377581c4d427f2b6fd0478eb283bb57692c5dd76a85338b0e8b2aadf47da254d6d2d5d49902007_mel_spectrograms.npy":
#         melDataDir = mels_folder+"/"+client
#         melDataDirSigned = mels_folder_signed+"/"+client
#         melDataDirCloned = mels_folder_cloned+"/"+client
#         originalMelData = normalizeMel(np.load(melDataDir))
#         signedMelData = normalizeMel(np.load(melDataDirSigned))
#         clonesMelData = normalizeMel(np.load(melDataDirCloned))
#         originalMelLabels = np.zeros(originalMelData.shape[0])
#         signedMelLabels = np.ones(signedMelData.shape[0])
#         clonesMelLabels = np.zeros(clonesMelData.shape[0])
#         if isFirst:
#             evalX = np.concatenate((originalMelData, signedMelData, clonesMelData), axis=0)
#             evalY = np.concatenate((originalMelLabels, signedMelLabels, clonesMelLabels), axis=0)
#             isFirst = False
#         evalX = np.concatenate((evalX, originalMelData, signedMelData, clonesMelData), axis=0)
#         evalY = np.concatenate((evalY, originalMelLabels, signedMelLabels, clonesMelLabels), axis=0)

# EER=dataEvaluateEER(verifyModelDir=verifyModelDir, melData=evalX, labels=evalY, device=torch.device("cuda:0"), isPrint=True)    

# print(accuracies)
# print(np.mean(np.array(accuracies)))

# verificationOnlyTesting()
print("EER Results for SecureSpectra (All) on VoxCeleb Dataset")
general_testing_func_EER()