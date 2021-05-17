import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple
import requests
from zipfile import ZipFile
INPUTVECTORSIZE = 300
BINARYVECTORSIZE = 128

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

# autoencoder class
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.latent = nn.Linear(INPUTVECTORSIZE, BINARYVECTORSIZE)
        self.ste = StraightThroughEstimator()
        self.decoder = nn.Linear(BINARYVECTORSIZE, INPUTVECTORSIZE)

    def forward(self, x):
        x = self.latent(x)
        x = self.ste(x)
        decoded = torch.tanh(self.decoder(x))

        return decoded

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 400000

    def __getitem__(self, index):
        # Select sample
        X = self.data[index]
        y = self.data[index]

        return X, y

# initialize model
model = AutoEncoder()

# initialize optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.95)

# loss function
lambdareg = 1.0/75.0 # divided by 75 because of batchsize
mse = nn.MSELoss()

#load data
f = open("glove.6b/glove.6b.300d.txt", "r", encoding="utf8")
lines = f.readlines()
words = []
embeddings: List[np.array] = []
for i, line in enumerate(lines):
    array = line.split()
    word = array[0]
    words.append(word)
    embedding = np.array(list(map(float, array[1:])))
    embeddings.append(embedding)

embeddings = torch.Tensor(embeddings)
data = Dataset(embeddings)
word2idx = {word: idx for idx, word in enumerate(words)}
idx2word = {idx: word for word, idx in word2idx.items()}
dataset = torch.utils.data.DataLoader(data, batch_size=75, shuffle=True)

# training loop
running_loss = 0
running_mseloss = 0
for epoch in range(25):
    print(epoch)
    for i, batch in enumerate(dataset):
        X, y = batch
        model.zero_grad()
        result = model(X)
        lrec = mse(result, y)
        loss = lrec
        #running_mseloss += lrec
        running_loss += loss
        loss.backward()
        optimizer.step()

        if i % 500 == 499:    # print loss after every 500 cycles

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

latent = model.latent(embeddings).detach().numpy()
binary = np.heaviside(latent, 0).astype(np.uint)

def diff(x, y):
    return abs(x - y) / BINARYVECTORSIZE

diffSums = []
for i, word in enumerate(words):
    if i > 99:
        break

    latentWord = latent[word2idx[word]]
    binaryWord = binary[word2idx[word]]
    diffSum = sum(list(map(diff, latentWord, binaryWord)))
    diffSums.append(diffSum)
    print(diffSum)

print(len(diffSums))
print('average: ' + str(sum(diffSums)/len(diffSums)))

## print to output file
#outputfilename = "ste" + str(BINARYVECTORSIZE) + ".txt"
#outputfile = open(outputfilename, "w", encoding="utf8")

#for word in words:
#    outputfile.write(word)
#    for dig in binary[word2idx[word]]:
#        outputfile.write(' ' + str(dig))
#
#    outputfile.write('\n')
#outputfile.close()
