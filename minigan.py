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




#generator = resnet18(pretrained=False, progress=True).cuda()
generator = MLPGEN().cuda()

#discriminator = get_model().cuda()
discriminator = MLP().cuda()
model = {"generator": generator, "discriminator": discriminator}
modelgen = {"generator": generator}
#model = {"discriminator": discriminator}
optimizer = {
    "generator": torch.optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999), weight_decay=0.1),
    "discriminator": torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999), weight_decay=0.1)
}
optimizergen = {
    "generator": torch.optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999), weight_decay=0.1)
}
optimizerdisc = {
    "discriminator": torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999), weight_decay=0.1)
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

loaders = {"train": loader_train, "valid": loader_val}
#loaders = {"train": loader_train}

torch.autograd.set_detect_anomaly(True)

resnetFA = torchvision.models.resnet18(pretrained=True).cuda()
resnetFA.fc = nn.Sequential()

def my_loss(output, target):
    #print(output.shape,'outputshape')
    #outx = output[:,:1000]
    #outy = output[:,-1000:]
    #gtx = target[:,:1000]
    #gty = target[:,-1000:]
    #print ('gt',gtx,gty)
    loss = torch.mean(torch.abs(output - target))
    #loss += torch.mean(torch.abs(gtx[:,:-1] - outx[:,1:]))
    #loss += torch.mean(torch.abs(gty[:,:-1] - outy[:,1:]))
    
    #loss += torch.mean(torch.abs(gtx[:,1:] - outx[:,:-1]))
    #loss += torch.mean(torch.abs(gty[:,1:] - outy[:,:-1]))
    return loss

class PretrainingGeneratorRunner(dl.Runner):
    def _handle_batch(self, batch):
        #print('batch',batch)
        batch_metrics = {}
        imagesStgGen = batch[0].detach().clone().cuda()
        sequencegtstggen = batch[1].detach().clone().cuda()
        generated_sequence = self.model["generator"](
            imagesStgGen)
        batch_metrics["loss_generator"] = \
            my_loss(generated_sequence, sequencegtstggen)
        try:
            batch_metrics["loss_generator"].backward()
            optimizergen['generator'].step()
        except:
            print('validation') 
               

        self.batch_metrics.update(**batch_metrics)


class PretrainingDiscriminatorRunner(dl.Runner):
    def _handle_batch(self, batch):
        imagesStgDis = batch[0].detach().clone().cuda()
        sequencegtstgdisc = batch[1].detach().clone().cuda()
        batch_metrics = {}

        #train discriminator
        generated_sequence = self.model['generator'](imagesStgDis)
        one_labels = torch.ones(128,1).cuda()
        zero_labels = torch.zeros(128,1).cuda()
        #hidden =  resnetFA(torch.cat([imagesStgDis[64:],imagesStgDis[64:]]))
        hidden =  torch.cat([imagesStgDis,imagesStgDis])
        h0, c0 = self.model['discriminator'].init_hidden(hidden)

        labels = torch.cat([
            one_labels, zero_labels
        ]).cuda()
        predictions = self.model["discriminator"](torch.cat([sequencegtstgdisc, generated_sequence])) #1s,0s
        batch_metrics["loss_discriminator"] = \
            F.binary_cross_entropy_with_logits(predictions, labels)
        try:
            batch_metrics["loss_discriminator"].backward()
            optimizerdisc['discriminator'].step()
        except:
            print('validation')

        self.batch_metrics.update(**batch_metrics)

class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):

        
        imagesStgGen = batch[0].detach().clone().cuda()
        imagesStgDis = batch[0].detach().clone().cuda()
        sequencegtstggen = batch[1].detach().clone().cuda()
        sequencegtstgdisc = batch[1].detach().clone().cuda()
        batch_metrics = {}
        
        #train discriminator fake
        generated_sequence = self.model['generator'](imagesStgDis)
        fake_labels = torch.zeros(128,1).cuda()
        hidden =  imagesStgDis
        h0, c0 = self.model['discriminator'].init_hidden(hidden)
        predictions = self.model["discriminator"](generated_sequence)
        batch_metrics["loss_discriminator"] = \
            F.binary_cross_entropy_with_logits(predictions, fake_labels)
        try:
            batch_metrics["loss_discriminator"].backward()
            optimizer['discriminator'].step()
        except:
            print('validation')
        
        #train discriminator real
        real_labels = torch.ones(128,1).cuda()
        hidden =  imagesStgDis
        h0, c0 = self.model['discriminator'].init_hidden(hidden)

        predictions = self.model["discriminator"](sequencegtstgdisc)
        batch_metrics["loss_discriminator"] = \
            F.binary_cross_entropy_with_logits(predictions, real_labels)

        try:
            batch_metrics["loss_discriminator"].backward()
            optimizer['discriminator'].step()
        except:
            print('validation')
        # Train the generator
        misleading_labels = torch.ones((64*2, 1)).cuda() #reward for fooling the discriminator

        generated_sequence = self.model["generator"](
            imagesStgGen) 
        hidden = imagesStgGen
        self.model['discriminator'].init_hidden(hidden)
        predictions = self.model["discriminator"](generated_sequence)
        #print("generator predictions ", predictions.shape)
        batch_metrics["loss_generator"] = \
            F.binary_cross_entropy_with_logits(predictions, misleading_labels)
        try:
            batch_metrics["loss_generator"].backward()
            optimizer['generator'].step()
        except:
            print('validation')
        
        self.batch_metrics.update(**batch_metrics)

runnergen = PretrainingGeneratorRunner()
runnergen.train(
    model=modelgen,
    optimizer=optimizergen,
    loaders=loaders,
    callbacks=None,
    main_metric="loss_generator",
    num_epochs=10,
    verbose=True,
    logdir="./logs_gan2"
)
runnerdisc = PretrainingDiscriminatorRunner()
runnerdisc.train(
    model=model,
    optimizer=optimizerdisc,
    loaders=loaders,
    callbacks=None,
    main_metric="loss_discriminator",
    num_epochs=10,
    verbose=True,
    logdir="./logs_gan2"
)


runner = CustomRunner()
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    callbacks=None,
    main_metric="loss_generator",
    num_epochs=10,
    verbose=True,
    logdir="./logs_gan2",
)
DonutDataset.displayCanvas('training-set.png', dataset_train, model['generator'])
DonutDataset.displayCanvas('validation-set.png', dataset_val, model['generator'])
