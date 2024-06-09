from securespectra_utils import *
import os

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)
    trainDevRatio = 9
    # mfcc_load = np.load("trainingMels_25k.npy")
    # labels_load = np.load("trainingY_25k.npy")
    # Path to the folder containing the .npy files
    datasetX_loc = "/home/obiwan/training_mels_librispeech/trainingMels.npy"
    datasetY_loc = "/home/obiwan/training_mels_librispeech/trainingY.npy"
    os.chdir("/home/obiwan/repos/watermarking_module/libri_experiments")
    datasetX = np.load(datasetX_loc)
    datasetY = np.load(datasetY_loc)
    numSamples = datasetY.shape[0]
    trainRatio = 0.9
    trainSamples = int(numSamples*trainRatio)
    mfcc_train = datasetX[:trainSamples]
    mfcc_val = datasetX[trainSamples:]
    labels_train = datasetY[:trainSamples]
    labels_val = datasetY[trainSamples:]
    
    
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
            torch.save(verifyModel.state_dict(), 'best_model_verify_only_libri.pt')
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