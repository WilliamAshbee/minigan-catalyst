import torch
from torch import nn


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
