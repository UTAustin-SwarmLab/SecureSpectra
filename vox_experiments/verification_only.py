from securespectra_utils import *
import os

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)
    trainDevRatio = 9
    # mfcc_load = np.load("trainingMels_25k.npy")
    # labels_load = np.load("trainingY_25k.npy")
    # Path to the folder containing the .npy files
    folder_path = '/home/obiwan/repos/watermarking_module/mel_spectograms/original_clients/'

    # Initialize an empty array
    accumulated_array = None

    # List all files in the folder and sort them (optional, for consistent order)
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

    # Iterate over the files
    for file_name in npy_files:
        # Load the current .npy file
        current_array = np.load(os.path.join(folder_path, file_name))
        
        # If the accumulated_array is None, it's the first file, so initialize it
        if accumulated_array is None:
            accumulated_array = current_array
        else:
            # Concatenate along the first dimension
            accumulated_array = np.concatenate((accumulated_array, current_array), axis=0)

    folder_path = '/home/obiwan/repos/watermarking_module/mel_spectograms/cloned_clients/'

    # List all files in the folder and sort them (optional, for consistent order)
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

    # Iterate over the files
    for file_name in npy_files:
        # Load the current .npy file
        current_array = np.load(os.path.join(folder_path, file_name))
        
        # If the accumulated_array is None, it's the first file, so initialize it
        if accumulated_array is None:
            accumulated_array = current_array
        else:
            # Concatenate along the first dimension
            accumulated_array = np.concatenate((accumulated_array, current_array), axis=0)
    
    
    # mfcc_load1 = np.concatenate((
    #     np.load("/home/obiwan/repos/watermarking_module/mel_spectograms/original_clients/0b42e481ca5fb9dc870188e2ff04095607aa1f2c3d58481350b9aa5be9748bac9337cdff30c1f0a48c79e7d01237f93192441f576afc842c8b33f642b4830561_mel_spectrograms.npy"),
    #     np.load("/home/obiwan/repos/watermarking_module/mel_spectograms/cloned_clients/0b42e481ca5fb9dc870188e2ff04095607aa1f2c3d58481350b9aa5be9748bac9337cdff30c1f0a48c79e7d01237f93192441f576afc842c8b33f642b4830561_mel_spectrograms.npy")),
    #                            axis=0)
    mfcc_load1 = accumulated_array
    labels_load1 = np.concatenate((np.ones(int(mfcc_load1.shape[0]/2)), np.zeros(int(mfcc_load1.shape[0]/2))), axis=0)
    rng = np.random.default_rng()  # Using default random generator for reproducibility

    # Shuffle X and y in unison
    indices = np.arange(mfcc_load1.shape[0])
    rng.shuffle(indices)
    mfcc_load= mfcc_load1[indices]
    labels_load = labels_load1[indices]
    
    
    numSample = mfcc_load.shape[0]
    numSampleDev = int(numSample/(trainDevRatio+1))
    numSampleTrain = numSample-numSampleDev
    mfcc_train = mfcc_load[:numSampleTrain,:,:]
    mfcc_val = mfcc_load[numSampleTrain:,:,:]
    labels_train = labels_load[:numSampleTrain]
    labels_val = labels_load[numSampleTrain:]
    
    
    train_dataset = evalDataset(mfcc_train,labels_train)
    val_dataset = evalDataset(mfcc_val,labels_val)
    # Define Data loaders
    train_loader = DataLoader(train_dataset,args.batch)
    valid_loader = DataLoader(val_dataset,args.batch)
    
    verifyModel = VerifierNet()
    verifyModel.apply(initialize_weights_xavier)
    verifyModel = verifyModel.to(device)
    
    optimizerVerify = optim.Adam(verifyModel.parameters(), lr=args.lr)
    criterionVerify = nn.BCELoss() 
    minimum_validation_loss_v = 99
    rounds_without_improvement_v = 0
    waiting_rounds = 10 
    
    verificationLossVal = []
    verificationLossTrain = []
    verificationAccVal = []
    verificationAccTrain = []
    for epoch in tqdm(range(args.epochs)):
        epochValLossVerify = 0
        epochTrainLossVerify = 0
        epochValAccVerify = 0
        epochTrainAccVerif = 0
        verifyModel.train()
        for i, (audios, labels) in enumerate(train_loader):
            audios = audios.to(device)
            labels = labels.to(device)
            predictions = verifyModel(audios).squeeze()
            loss = criterionVerify(predictions, labels)
            optimizerVerify.zero_grad()
            loss.backward()
            optimizerVerify.step()
            epochTrainLossVerify += loss.item()
            epochTrainAccVerif+=calculate_accuracy(predictions, labels)
        epochTrainLossVerify=epochTrainLossVerify/(i+1)
        epochTrainAccVerif=epochTrainAccVerif/(i+1)
        print(f"[Training Loss: {epochTrainLossVerify}]")
        print(f"[Training Accuracy: {epochTrainAccVerif}]")
        verificationLossTrain.append(epochTrainLossVerify)
        verificationAccTrain.append(epochTrainAccVerif)
        
        
        verifyModel.eval() 
        
        for i_v, (audios_v, labels_v) in enumerate(valid_loader):
            audios_v = audios_v.to(device)
            labels_v = labels_v.to(device)
            predictions_v = verifyModel(audios_v).squeeze()
            loss_v = criterionVerify(predictions_v, labels_v)
            epochValLossVerify += loss_v.item()
            epochValAccVerify+=calculate_accuracy(predictions_v, labels_v)
        
        epochValLossVerify=epochValLossVerify/(i_v+1)
        epochValAccVerify=epochValAccVerify/(i_v+1)
        print(f"[Validation Loss: {epochValLossVerify}]")
        print(f"[Validation Accuracy: {epochValAccVerify}]")
        verificationLossVal.append(epochValLossVerify)
        verificationAccVal.append(epochValAccVerify)   

        if epochValLossVerify < minimum_validation_loss_v:
            minimum_validation_loss_v = epochValLossVerify
            rounds_without_improvement_v = 0
            torch.save(verifyModel.state_dict(), 'best_model_verify_only.pt')
        else:
            rounds_without_improvement_v += 1
        
        if rounds_without_improvement_v == waiting_rounds: # or rounds_without_improvement_s == waiting_rounds or
            print("No improvement for {} rounds. Early stopping...".format(waiting_rounds))
            break

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch",
        default=256,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=1000,
        type=int,
        help="number of epochs",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="./data",
        type=str,
        help="Repository containing the dataset",
    )
    parser.add_argument(
        "-l",
        "--lr",
        default=0.0002,
        type=float,
        help="The learning rate of the training",
    )
    args = parser.parse_args()
    os.chdir("/home/obiwan/repos/watermarking_module")
    main(args)