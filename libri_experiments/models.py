import torch
from torch import nn
import torch.nn.functional as F
from utilities import crop_center
def replicate_and_concat(unet_feature_map, message_vector=torch.tensor([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])):
    N, C, H, W = unet_feature_map.size()  # Extract dimensions of x5
    # Assuming message_vector is a 1D tensor of shape [128]
    
    # Reshape message_vector to [1, 128, 1, 1] to make it compatible for replication
    message_vector = message_vector.view(1, -1, 1, 1)
    
    # Replicate message_vector to match the spatial dimensions of x5
    # Replication: 1x1 to H/16xW/16
    replicated_message = message_vector.repeat(N, 1, H, W)
    replicated_message = replicated_message.to("cuda:0")
    
    # Concatenate replicated_message with x5 along the channel dimension
    result = torch.cat((unet_feature_map, replicated_message), dim=1)  # Resulting tensor dimensions: [N, 512+128, H/16, W/16]
    
    return result

class ASPPModule(nn.Module):

    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        )
        self.conv2 = Conv2DBNActiv(
            nin, nout, 1, 1, 0, activ=activ
        )
        self.conv3 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[0], dilations[0], activ=activ
        )
        self.conv4 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[1], dilations[1], activ=activ
        )
        self.conv5 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[2], dilations[2], activ=activ
        )
        self.bottleneck = Conv2DBNActiv(
            nout * 5, nout, 1, 1, 0, activ=activ
        )
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.bottleneck(out)

        if self.dropout is not None:
            out = self.dropout(out)

        return out


class LSTMModule(nn.Module):

    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super(LSTMModule, self).__init__()
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        self.lstm = nn.LSTM(
            input_size=nin_lstm,
            hidden_size=nout_lstm // 2,
            bidirectional=True
        )
        self.dense = nn.Sequential(
            nn.Linear(nout_lstm, nin_lstm),
            nn.BatchNorm1d(nin_lstm),
            nn.ReLU()
        )

    def forward(self, x):
        N, _, nbins, nframes = x.size()
        h = self.conv(x)[:, 0]  # N, nbins, nframes
        h = h.permute(2, 0, 1)  # nframes, N, nbins
        h, _ = self.lstm(h)
        h = self.dense(h.reshape(-1, h.size()[-1]))  # nframes * N, nbins
        h = h.reshape(nframes, N, 1, nbins)
        h = h.permute(1, 2, 3, 0)

        return h
    
class Conv2DBNActiv(nn.Module):

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin, nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(nout),
            activ()
        )

    def __call__(self, x):
        return self.conv(x)
    
class Conv2DBNActivTransposed(nn.Module):

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActivTransposed, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                nin, nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(nout),
            activ()
        )

    def __call__(self, x):
        return self.conv(x)


class Encoder(nn.Module):

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        return h


class Decoder(nn.Module):

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False):
        super(Decoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        # self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        x0 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        if skip is not None:
            # skip = crop_center(skip, x)
            x1 = torch.cat([x0, skip], dim=1)

        h = self.conv1(x1)
        # h = self.conv2(h)

        if self.dropout:
            h = self.dropout(h)

        return h

class SignatureNet(nn.Module):

    def __init__(self, nin=1, nout=16):
        super(SignatureNet, self).__init__()
        self.enc0 = Conv2DBNActiv(nin, nin, 3, 1, 1) # 1 x 128 x 235, 16 x 64 (nchxnfeat) 1x16x128x235
        self.enc1 = Conv2DBNActiv(nin, nout, 3, 2, 1) # 1 x 128 x 235, 16 x 64 (nchxnfeat) 1x16x128x235
        self.enc2 = Encoder(nout, nout * 2, 3, 2, 1) # 16 x 64, 32 x 32
        self.enc3 = Encoder(nout * 2, nout * 4, 3, 2, 1) # 32 x 32, 64 x16
        self.enc4 = Encoder(nout * 4, nout * 8, 3, 2, 1) # 64 x 16, 128 x 8

        self.deepest = Encoder(nout * (8+2), nout * 4, 3, 1, 1) # 160x8, 64 x 8
        #self.aspp = ASPPModule(nout * 8, nout * 8, dilations, dropout=True) # TODO add the concatination function here, # 

        # self.dec4 = Decoder(nout * (8 + 8), nout * 4, 3, 1, 1) # 128 x 8  + 128 x 8 = 256x8 -> 64 x 16
        # self.dec3 = Decoder(nout * (4 + 4), nout * 2, 3, 1, 1)
        # self.dec2 = Decoder(nout * (2 + 2), nout * 1, 3, 1, 1)
        # self.dec1 = Decoder(nout * (1 + 1), nin * 1, 3, 1, 1)
        self.dec4 = Decoder(nout * 8, nout * 2, 3, 1, 1) # 128 x 8  + 128 x 8 = 256x8 -> 64 x 16
        self.dec3 = Decoder(nout * 4, nout * 1, 3, 1, 1)
        self.dec2 = Decoder(nout * 2, nout * 1, 3, 1, 1)
        self.dec1 = Decoder(nout + nin, nin * 1, 3, 1, 1)

    def __call__(self, x): # , private_key
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # add the message
        h1 = replicate_and_concat(unet_feature_map=e4) #, message_vector=private_key 160x8x16
        # resize it with a convolution
        h2 = self.deepest(h1) #, message_vector=private_key 128x8x16
        # h = self.aspp(e4) # TODO change this function to take your message as well h = self.aspp(e4,m)
        # # add the message and resize the box
        h3 = self.dec4(h2, e3)
        h4 = self.dec3(h3, e2)
        h5 = self.dec2(h4, e1)
        h = self.dec1(h5, e0)
        # print("a")
        return h
    
class VerifierNet(nn.Module):
    def __init__(self, final_input_size=256*2, output_size=1) -> None:
        super(VerifierNet, self).__init__()
        self.conv1 = Conv2DBNActiv(1, 4, 3, 2, 1)
        self.conv2 = Conv2DBNActiv(4, 8, 3, 2, 1)
        self.conv3 = Conv2DBNActiv(8, 16, 3, 2, 1) 
        self.conv4 = Conv2DBNActiv(16, 32, 3, 2, 1) 
        self.conv5 = Conv2DBNActiv(32, 64, 3, 2, 1) 
        self.conv6 = Conv2DBNActiv(64, 128, 3, 2, 1) 
        self.conv7 = Conv2DBNActiv(128, 256, 3, 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=final_input_size, out_features=output_size)
        self.sigmoid = nn.Sigmoid()  
     
    def __call__(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = F.relu(self.conv6(x5))
        x7 = F.relu(self.conv7(x6))
        # Flatten the output for the fully connected layer
        x8 = x7.view(x7.size(0), -1)  # Flatten while keeping the batch size dimension
        # Apply the fully connected layer
        x9 = self.fc1(x8)
        x10 = self.sigmoid(x9)
        # print(torch.isnan(x10).any())
        return x10
## class BaseNet(nn.Module):

#     def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
#         super(BaseNet, self).__init__()
#         self.enc1 = Conv2DBNActiv(nin, nout, 3, 1, 1)
#         self.enc2 = Encoder(nout, nout * 2, 3, 2, 1)
#         self.enc3 = Encoder(nout * 2, nout * 4, 3, 2, 1)
#         self.enc4 = Encoder(nout * 4, nout * 6, 3, 2, 1)
#         self.enc5 = Encoder(nout * 6, nout * 8, 3, 2, 1)

#         self.aspp = ASPPModule(nout * 8, nout * 8, dilations, dropout=True)

#         self.dec4 = Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
#         self.dec3 = Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
#         self.dec2 = Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
#         self.lstm_dec2 = LSTMModule(nout * 2, nin_lstm, nout_lstm) # I do not need this layer and it seems to me the ordering is wrong for concatination
#         self.dec1 = Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

#     def __call__(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         e5 = self.enc5(e4)

#         h = self.aspp(e5)

#         h = self.dec4(h, e4)
#         h = self.dec3(h, e3)
#         h = self.dec2(h, e2)
#         h = torch.cat([h, self.lstm_dec2(h)], dim=1)
#         h = self.dec1(h, e1)

#         return h