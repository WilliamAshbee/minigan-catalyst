import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.contrib.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.nn.modules import Flatten, GlobalMaxPool2d, Lambda
from resnet import resnet18, resnext101_32x8d
from DonutDataset import DonutDataset
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.autograd import Variable
import torchvision
import numpy as np
from mlp import MLP
from mlpgen import MLPGEN
cuda = True
latent_dim = 128


##############lstm
## Multilayer LSTM based classifier taking in 200 dimensional fixed time series inputs
## Multilayer LSTM based classifier taking in 200 dimensional fixed time series inputs
class LSTMClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size):
        super(LSTMClassifier, self).__init__()
        self.arch = 'lstm'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_dir = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.h0 = None
        self.c0 = None

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim*self.num_dir, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )

        self.hidden = self.init_hidden()

    def setH0(self, h0):
      self.h0 = h0

    def get_hidden(self):
      return self.init_hidden(self.h0)

    def init_hidden(self, h0=None):
        if h0 is None:
            if cuda:
                h0 = Variable(torch.zeros(self.num_layers*self.num_dir,
                                            self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers*self.num_dir,
                                            self.batch_size, self.hidden_dim).cuda())
                self.h0 = h0
                self.c0 = c0
            else:
                h0 = Variable(torch.zeros(self.num_layers *
                                            self.num_dir, self.batch_size, self.hidden_dim))
                c0 = Variable(torch.zeros(self.num_layers *
                                            self.num_dir, self.batch_size, self.hidden_dim))
                self.h0 = h0
                self.c0 = c0

        else:
            if cuda:
                print('seeding')
                if h0.shape[0]!=6:
                    #b = torch.randn(128,512)
                    b = h0.cpu().numpy()
                    a = torch.from_numpy(np.tile(b, (6, 1, 1))).cuda()
                    self.h0 = a
                else:
                    self.h0 = h0
                c0 = Variable(torch.zeros(self.num_layers*self.num_dir,
                                          self.batch_size, self.hidden_dim).cuda())
                self.c0 = c0
            else:
                self.h0 = h0
                c0 = Variable(torch.zeros(self.num_layers *
                                          self.num_dir, self.batch_size, self.hidden_dim))
                self.c0 = c0

        return (self.h0, self.c0)

    def forward(self, x):  # x is (batch_size, 1, 200), permute to (200, batch_size, 1)
        x = x.permute(2, 0, 1)
        # See: https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/2
        if self.h0 is None:
          self.init_hidden()
        lstm_out, (h, c) = self.lstm(x, self.get_hidden())
        y = self.hidden2label(lstm_out[-1])
        #self.h0 = None
        return y

    def setBatchSize(self, batch_size=1):
        self.batch_size = batch_size


def get_model():  # tuples of (batch_size, model)
    return LSTMClassifier(
        in_dim=2,
        hidden_dim=512,
        num_layers=3,
        dropout=0.8,
        bidirectional=True,
        num_classes=1,  # bce loss for discriminator
        batch_size=128
    )

##############lstm


#generator = resnet18(pretrained=False, progress=True).cuda()
generator = MLPGEN().cuda()

#discriminator = get_model().cuda()
discriminator = MLP().cuda()
model = {"generator": generator, "discriminator": discriminator}
#model = {"discriminator": discriminator}
optimizer = {
    "generator": torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999)),
    "discriminator": torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))
}
#loaders = {
#    "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
#}

dataset_train = DonutDataset(128*10)
dataset_val = DonutDataset(128*2)

mini_batch = 128
loader_train = data.DataLoader(
    dataset_train, batch_size=mini_batch,
    sampler=RandomSampler(data_source=dataset_train),
    num_workers=4)

loader_val = data.DataLoader(
    dataset_val, batch_size=mini_batch,
    sampler=RandomSampler(data_source=dataset_val),
    num_workers=4)

#loaders = {"train": loader_train, "valid": loader_val}
loaders = {"train": loader_train}

torch.autograd.set_detect_anomaly(True)

resnetFA = torchvision.models.resnet18(pretrained=True).cuda()
resnetFA.fc = nn.Sequential()


class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):

        imagesStgGen = batch[0].detach().clone().cuda()
        imagesStgDis = batch[0].detach().clone().cuda()
        #print('images',images.shape)
        sequencegtstggen = batch[1].detach().clone().cuda()
        sequencegtstgdisc = batch[1].detach().clone().cuda()
        #real_images, _ = batch
        batch_metrics = {}
        if batch[0].shape[0] != 128:
            print("this is wrong!!!", batch[0].shape)
            #images = torch.cat([images,images,images,images,images,images,images,images])
            #sequence = torch.cat([sequence,sequence,sequence,sequence,sequence,sequence,sequence,sequence])

        #train discriminator fake
        generated_sequence = self.model['generator'](imagesStgDis)
        #generated_sequence = generated_sequence.reshape(-1, 2, 1000)

        fake_labels = torch.zeros(128,1).cuda()

        #hidden =  resnetFA(torch.cat([imagesStgDis[64:],imagesStgDis[64:]]))
        hidden =  imagesStgDis
        h0, c0 = self.model['discriminator'].init_hidden(hidden)

        predictions = self.model["discriminator"](generated_sequence)
        batch_metrics["loss_discriminator"] = \
            F.binary_cross_entropy_with_logits(predictions, fake_labels)

        batch_metrics["loss_discriminator"].backward()
        optimizer['discriminator'].step()

        real_labels = torch.ones(128,1).cuda()

        #hidden =  resnetFA(torch.cat([imagesStgDis[64:],imagesStgDis[64:]]))
        hidden =  imagesStgDis
        h0, c0 = self.model['discriminator'].init_hidden(hidden)

        predictions = self.model["discriminator"](sequencegtstgdisc)
        batch_metrics["loss_discriminator"] = \
            F.binary_cross_entropy_with_logits(predictions, real_labels)

        batch_metrics["loss_discriminator"].backward()
        optimizer['discriminator'].step()

        # Train the generator
        misleading_labels = torch.zeros((64*2, 1)).cuda()

        generated_sequence = self.model["generator"](
            imagesStgGen)  # this needs to be redone !!!!!!!
        #generated_sequence = generated_sequence.reshape(-1, 2, 1000)

        # resnetFA(generated_sequence))
        #hidden =  resnetFA(imagesStgGen)
        hidden = imagesStgGen
        self.model['discriminator'].init_hidden(hidden)
        predictions = self.model["discriminator"](generated_sequence)
        #print("generator predictions ", predictions.shape)
        batch_metrics["loss_generator"] = \
            F.binary_cross_entropy_with_logits(predictions, misleading_labels)
        batch_metrics["loss_generator"].backward()
        optimizer['generator'].step()

        #batch_metrics["loss_generator"].step()
        #print("batchmetrics",str(**batch_metrics))
        self.batch_metrics.update(**batch_metrics)


runner = CustomRunner()
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    callbacks=None,
    main_metric="loss_generator",
    num_epochs=40,
    verbose=True,
    logdir="./logs_gan2",
)

DonutDataset.displayCanvas(dataset_train, model['generator'])
