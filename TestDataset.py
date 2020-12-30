import torch
import numpy as np
import pylab as plt
from skimage import filters
import math

side = 32
numpoints = 32
def plot_all( sample = None, labels = None):
    img = sample[0,:,:].squeeze().cpu().numpy()
    plt.imshow(img, cmap=plt.cm.gray_r)
    #print(labels.shape)

    X = labels[:numpoints]
    Y = labels[-numpoints:]
    s = [.001 for x in range(numpoints)]
    c = ['red' for x in range(numpoints)]
    print ('sumx', torch.sum(X))
    print ('sumy', torch.sum(Y))
    ascatter = plt.scatter(Y.cpu().numpy(),X.cpu().numpy(),s = s,c = c)
    plt.gca().add_artist(ascatter)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, length = 10):
        self.length = length
        self.values = {}
        self.values['canvas'] = torch.zeros(length,32,32)
        self.values['points'] = torch.ones(length,2*numpoints)*31.0
        for i in range(length):
            value = np.random.randint(32)
            self.values['canvas'][i,:,value] = 1
            #self.values['points'][i,] 
        self.values['canvas']

        assert self.values['canvas'].shape[0] == self.length
        assert self.values['points'].shape[0] == self.length

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        canvas = self.values["canvas"]
        
        canvas = canvas[idx,:,:]
        assert canvas.shape == (side,side)
        canvas = torch.reshape(canvas,(1,side,side))
        assert canvas.shape == (1,side,side)
        
        #canvas = torch.from_numpy(canvas)
        canvas = canvas.repeat(3, 1, 1).float()
        assert canvas.shape == (3,side,side)
        points = self.values["points"]
        points = points[idx,:]

        #print('points', points.shape)
        return canvas, points
    
    @staticmethod
    def displayCanvas(title,dataset):
        #model.setBatchSize(batch_size = 1)
        for i in range(100):
            sample, labels = dataset[i]
            plt.subplot(10,10,i+1)
            plot_all(sample = sample, labels = labels)
            plt.axis('off')
        plt.savefig(title,dpi=600)

dataset = TestDataset(length = 100)
TestDataset.displayCanvas('finalfig.png',dataset)
