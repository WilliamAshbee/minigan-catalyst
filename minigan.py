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
cuda = True
latent_dim = 128

##############lstm
## Multilayer LSTM based classifier taking in 200 dimensional fixed time series inputs
class LSTMClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size):
        super(LSTMClassifier, self).__init__()
        self.arch = 'lstm'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_dir = 2 if bidirectional else 1
        self.num_layers = num_layers

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

    def init_hidden(self):
        if cuda:
            h0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x): # x is (batch_size, 1, 200), permute to (200, batch_size, 1)
        #print("permute",x.shape)
        x = x.permute(2, 0, 1)
        # See: https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/2
        lstm_out, (h, c) = self.lstm(x, self.init_hidden())
        y  = self.hidden2label(lstm_out[-1])
        return y


def get_model(): # tuples of (batch_size, model)
    return LSTMClassifier(
        in_dim=2,
        hidden_dim=120,
        num_layers=3,
        dropout=0.8,
        bidirectional=True,
        num_classes=1,#bce loss for discriminator
        batch_size=256
    )

##############lstm



generator = resnet18(pretrained=False, progress=True).cuda()

discriminator = get_model().cuda()

model = {"generator": generator, "discriminator": discriminator}
optimizer = {
    "generator": torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999)),
    "discriminator": torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999)),
}
#loaders = {
#    "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
#}

dataset_train = DonutDataset(400)
dataset_val = DonutDataset(256)

mini_batch = 256
loader_train = data.DataLoader(
    dataset_train, batch_size=mini_batch,
    sampler=RandomSampler(data_source = dataset_train),
    num_workers=4)

loader_val = data.DataLoader(
    dataset_val, batch_size=mini_batch,
    sampler=RandomSampler(data_source = dataset_val),
    num_workers=4)

#loaders = {"train": loader_train, "valid": loader_val}
loaders = {"train": loader_train}


class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        images = batch[0].cuda()
        sequence = batch[1].cuda()
        #real_images, _ = batch
        batch_metrics = {}
        
        # Sample random points in the latent space
        #batch_size = real_images.shape[0]
        #random_latent_vectors = torch.randn(batch_size, latent_dim).to(self.device)
        
        # Decode them to fake images
        #generated_images = self.model["generator"](random_latent_vectors).detach()
        #print('batch',batch[0])
        #print('batch',batch[1])
        generated_sequence = self.model['generator'](images[128:])
        generated_sequence = generated_sequence.reshape(-1,2,1000)
        print(generated_sequence.shape)
        print(batch[1].shape)
        print('+-+-+-\n')

        combined_sequence = torch.cat([generated_sequence,sequence[128:]], axis = 0).cuda()
        labels = torch.zeros((256,1)).cuda()
        labels[-128] = 1
        # Combine them with real images
        #combined_images = torch.cat([generated_images, real_images])
        
        # Assemble labels discriminating real from fake images
        #labels = torch.cat([
        #    torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))
        #]).to(self.device)
        # Add random noise to the labels - important trick!
        #labels += 0.05 * torch.rand(labels.shape).to(self.device)
        
        # Train the discriminator
        predictions = self.model["discriminator"](combined_sequence)
        batch_metrics["loss_discriminator"] = \
          F.binary_cross_entropy_with_logits(predictions, labels)
        
        # Sample random points in the latent space
        #random_latent_vectors = torch.randn(batch_size, latent_dim).to(self.device)
        # Assemble labels that say "all real images"
        misleading_labels = torch.zeros((128*2, 1)).cuda()
        
        # Train the generator
        generated_sequence = self.model["generator"](images) #this needs to be redone !!!!!!!
        generated_sequence = generated_sequence.reshape(-1,2,1000)
        predictions = self.model["discriminator"](generated_sequence)
        batch_metrics["loss_generator"] = \
          F.binary_cross_entropy_with_logits(predictions, misleading_labels)
        
        self.batch_metrics.update(**batch_metrics)

runner = CustomRunner()
runner.train(
    model=model, 
    optimizer=optimizer,
    loaders=loaders,
    callbacks=[
        dl.OptimizerCallback(
            optimizer_key="generator", 
            metric_key="loss_generator"
        ),
        dl.OptimizerCallback(
            optimizer_key="discriminator", 
            metric_key="loss_discriminator"
        ),
    ],
    main_metric="loss_generator",
    num_epochs=3,
    verbose=True,
    logdir="./logs_gan2",
)

