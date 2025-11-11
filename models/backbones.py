import math
import torch
import torch.nn as nn

# Adapted from https://github.com/yuvenduan/PlasticRNNs/blob/master/models/cnns/R2D2_embedding.py

# Adapted from Meta-Learning with Differentiable Convex Optimization, CVPR 2019
# https://github.com/kjunelee/MetaOptNet


# Embedding network used in Meta-learning with differentiable closed-form solvers
# (Bertinetto et al., in submission to NIPS 2018).
# They call the ridge rigressor version as "Ridge Regression Differentiable Discriminator (R2D2)."
  
# Note that they use a peculiar ordering of functions, namely conv-BN-pooling-lrelu,
# as opposed to the conventional one (conv-BN-lrelu-pooling).
  
def R2D2_conv_block(in_channels, out_channels, retain_activation=True, keep_prob=1.0):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2)
    )
    if retain_activation:
        block.add_module("LeakyReLU", nn.LeakyReLU(0.1))

    if keep_prob < 1.0:
        block.add_module("Dropout", nn.Dropout(p=1 - keep_prob, inplace=False))

    return block

class R2D2Embedding(nn.Module):
    def __init__(self, x_dim=3, h1_dim=96, h2_dim=192, h3_dim=384, z_dim=512, \
                 retain_last_activation=False):
        super(R2D2Embedding, self).__init__()

        self.block1 = R2D2_conv_block(x_dim, h1_dim)
        self.block2 = R2D2_conv_block(h1_dim, h2_dim)
        self.block3 = R2D2_conv_block(h2_dim, h3_dim, keep_prob=0.9)
        # In the last conv block, we disable activation function to boost the classification accuracy.
        # This trick was proposed by Gidaris et al. (CVPR 2018).
        # With this trick, the accuracy goes up from 50% to 51%.
        # Although the authors of R2D2 did not mention this trick in the paper,
        # we were unable to reproduce the result of Bertinetto et al. without resorting to this trick.
        self.block4 = R2D2_conv_block(h3_dim, z_dim, retain_activation=retain_last_activation, keep_prob=0.7)
  
    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        # Flatten and concatenate the output of the 3rd and 4th conv blocks as proposed in R2D2 paper.
        return torch.cat((b3.view(b3.size(0), -1), b4.view(b4.size(0), -1)), 1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, retain_activation=True):
        super(ConvBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if retain_activation:
            self.block.add_module("ReLU", nn.ReLU(inplace=True))
        self.block.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        
    def forward(self, x):
        out = self.block(x)
        return out

# Embedding network used in Matching Networks (Vinyals et al., NIPS 2016), Meta-LSTM (Ravi & Larochelle, ICLR 2017),
# MAML (w/ h_dim=z_dim=32) (Finn et al., ICML 2017), Prototypical Networks (Snell et al. NIPS 2017).

class ProtoNetEmbedding(nn.Module):
    def __init__(self, x_dim=3, h_dim=64, z_dim=64, retain_last_activation=True):
        super(ProtoNetEmbedding, self).__init__()
        self.encoder = nn.Sequential(
          ConvBlock(x_dim, h_dim),
          ConvBlock(h_dim, h_dim),
          ConvBlock(h_dim, h_dim),
          ConvBlock(h_dim, z_dim, retain_activation=retain_last_activation),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)