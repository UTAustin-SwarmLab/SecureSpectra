from securespectra_utils import *

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)
    datasetX_loc = "/home/obiwan/training_mels_vox/trainingMels.npy"
    datasetY_loc = "/home/obiwan/training_mels_vox/trainingY.npy"
    os.chdir("/home/obiwan/repos/watermarking_module/vox_experiments")
    datasetX = np.load(datasetX_loc)
    datasetY = np.load(datasetY_loc)
    numSamples = datasetY.shape[0]
    trainRatio = 0.9
    trainSamples = int(numSamples*trainRatio)
    
    eval_X = datasetX[trainSamples:]
    eval_Y = datasetY[trainSamples:]
    datasetX = datasetX[:trainSamples]
    datasetY = datasetY[:trainSamples]
    
    # train_dataset = signatureDataset("/home/obiwan/repos/models/original_CV/melspecs.npy")
    # val_dataset = signatureDataset("/home/obiwan/repos/models/original_CV_val/melspecs.npy")
    # Define Data loaders
    
    train_dataset = evalDataset(datasetX, datasetY)
    val_dataset = evalDataset(eval_X, eval_Y)
    
    train_loader = DataLoader(train_dataset,args.batch)
    valid_loader = DataLoader(val_dataset,args.batch)
    
    # Define Models
    signModel = SignatureNet()
    signModel.apply(initialize_weights_xavier)
    signModel = signModel.to(device)
    verifyModel = VerifierNet()
    verifyModel.apply(initialize_weights_xavier)
    verifyModel = verifyModel.to(device)
    # Define optimizers
    optimizerSign = optim.Adam(signModel.parameters(), lr=args.lr)
    optimizerVerify = optim.Adam(verifyModel.parameters(), lr=args.lr/10)
    
    # Define the losses of each module
    criterionVerify = nn.BCELoss() 
    criterionSign = nn.L1Loss()
    
    # TODO a function to load the state dictionaries    
    # if (args.weights != "None"):
    #     print(f"Load weights from {args.weights}")
    #     models.load(args.weights)
    # ************ training starts here ************
    # rounds_without_improvement = 0
    minimum_validation_loss = 99
    minimum_validation_loss_v = 99
    rounds_without_improvement_v = 0
    waiting_rounds = 10 
    initialization_rounds = 20
    verificationLossVal = []
    signatureLossVal = []
    verificationLossTrain = []
    signatureLossTrain = []
    verificationAccuracyVal = []
    for epoch in tqdm(range(args.epochs)):
        epochValLossSign = 0
        epochTrainLossSign = 0
        epochValLossVerify = 0
        epochTrainLossVerify = 0
        epochValAccVerify = 0
        epochTrainAccVerif = 0
        signModel.train()
        verifyModel.train()
        
        for i, (audios, label) in enumerate(train_loader):

            # print(i)
            # Ground Truths for Models            
            signedLabels = torch.ones(audios.size(0), 1, requires_grad=False).to(device)
            unsignedLabels = torch.zeros(audios.size(0), 1, requires_grad=False).to(device)
            audios = audios.to(device)
            
            signedAudios = signModel(audios)
            signLoss = criterionSign(audios,signedAudios)
            
            # signLoss.backward(retain_graph=True)
            
            if epoch > initialization_rounds:
                signedAudiosPredictions = verifyModel(signedAudios)
                unsignedAudiosPredictions = verifyModel(audios)
                unsignedImageLoss = criterionVerify(unsignedAudiosPredictions,unsignedLabels)
                signedImageLoss = criterionVerify(signedAudiosPredictions,signedLabels)
                verifyLoss = (unsignedImageLoss + signedImageLoss)/2
                totalLoss = verifyLoss + signLoss
            else:
                totalLoss = signLoss
                
            optimizerSign.zero_grad() 
            optimizerVerify.zero_grad()
            totalLoss.backward()
            
            optimizerSign.step()
            # # Inspect gradients
            # for name, param in signModel.named_parameters():
            #     print(f'Parameter: {name}, Gradient norm: {param.grad.norm()}')
            if epoch > initialization_rounds:
                optimizerVerify.step()
                epochTrainLossVerify += verifyLoss.item()
                # Print progress
            epochTrainLossSign += signLoss.item()
        epochTrainLossSign = epochTrainLossSign/(i+1)
        print(f"[Epoch {epoch}/{args.epochs}] [Signature loss: {epochTrainLossSign}]")
        signatureLossTrain.append(epochTrainLossSign)
        if epoch > initialization_rounds:
            epochTrainLossVerify=epochTrainLossVerify/(i+1)
            print(f"[Epoch {epoch}/{args.epochs}] [Verification loss: {epochTrainLossVerify}]")
            verificationLossTrain.append(epochTrainLossVerify)
        signModel.eval()
        verifyModel.eval()
        
        for i_v, (audios_v, label_v) in enumerate(valid_loader):
            # print(i_v)
            # validation only for the signature model
            audios_v = audios_v.to(device)
            signedAudios = signModel(audios_v)
            signLoss = criterionSign(audios_v,signedAudios)
            epochValLossSign+=signLoss.item()
            
            if epoch>initialization_rounds:
                signedLabels = torch.ones(audios_v.size(0), 1, requires_grad=False).to(device)
                unsignedLabels = torch.zeros(audios_v.size(0), 1, requires_grad=False).to(device)
                signedAudiosPredictions = verifyModel(signedAudios)
                unsignedAudiosPredictions = verifyModel(audios_v)
                unsignedImageLoss = criterionVerify(unsignedAudiosPredictions,unsignedLabels)
                signedImageLoss = criterionVerify(signedAudiosPredictions,signedLabels)
                verifyLoss = (unsignedImageLoss.item() + signedImageLoss.item())/2
                epochValLossVerify += verifyLoss
                epochValAccVerify+=(calculate_accuracy(signedAudiosPredictions, signedLabels)+calculate_accuracy(unsignedAudiosPredictions, unsignedLabels))/2
                                
        epochValLossSign=epochValLossSign/(i_v+1)
        print(f"[Epoch {epoch}/{args.epochs}][Validation Signature loss: {epochValLossSign}]")
        signatureLossVal.append(epochValLossSign)
        
        if epoch>initialization_rounds:
            epochValLossVerify=epochValLossVerify/(i_v+1)
            epochValAccVerify=epochValAccVerify/(i_v+1)
            print(f"[Epoch {epoch}/{args.epochs}] [Validation Verification loss: {epochValLossVerify}]")
            print(f"[Epoch {epoch}/{args.epochs}] [Verification Accuracy: {epochValAccVerify}]")
            verificationLossVal.append(epochValLossSign)
            verificationAccuracyVal.append(epochValAccVerify)
        # Check if validation loss has improved and save if it does not improve for a waiting_num_rounds    
        if epochValLossSign < minimum_validation_loss:
            minimum_validation_loss = epochValLossSign
            rounds_without_improvement_s = 0
            if epoch>initialization_rounds:
                # Save the model
                torch.save(signModel.state_dict(), 'best_model_sign_vox.pt')
                torch.save(verifyModel.state_dict(), 'best_model_verify_vox.pt')
        else:
            rounds_without_improvement_s += 1
            
        if epoch>initialization_rounds:
            if epochValLossVerify<minimum_validation_loss_v:
                minimum_validation_loss_v = epochValLossVerify
                rounds_without_improvement_v = 0
                torch.save(signModel.state_dict(), 'best_model_sign_vox.pt')
                torch.save(verifyModel.state_dict(), 'best_model_verify_vox.pt')
            else:
                rounds_without_improvement_v+=1


        # Check if we need to early stop
        if rounds_without_improvement_v == waiting_rounds: # or rounds_without_improvement_s == waiting_rounds or
            print("No improvement for {} rounds. Early stopping...".format(waiting_rounds))
            break
        
    save_history(verificationLossVal,signatureLossVal,verificationLossTrain,signatureLossTrain,verificationAccuracyVal,target_path="history_libri.json")
        


    
if __name__ == '__main__':
    # DONE add validation for signature only training DONE
    # DONE add validation for verification too DONE
    # DONE jointly train both of them to get the final model DONE
    # DONE write the code only verification model without signature
    # get the datasets for all cases signed, unsigned, deepfake
    # get the different datasets
    # get the different baselines
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