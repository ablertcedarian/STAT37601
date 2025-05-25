import h5py
import torch
import numpy as np
# Torch functions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Utility to track progress of a routine.
from tqdm.notebook import trange, tqdm

# Folder with data
predir=''
datadir='data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_mnist(train_size=50000):


    data=np.float64(np.load(datadir+'/MNIST_data.npy'))
    labels=np.float32(np.load(datadir+'/MNIST_labels.npy'))
    print('Mnist data shape',data.shape)
    data=np.float32(data)/255.
    train_dat=data[0:train_size].reshape((-1,1,28,28))
    train_labels=np.int32(labels[0:train_size])
    val_dat=data[train_size:60000].reshape((-1,1,28,28))
    val_labels=np.int32(labels[train_size:60000])
    test_dat=data[60000:70000].reshape((-1,1,28,28))
    test_labels=np.int32(labels[60000:70000])

    return (train_dat, train_labels), (val_dat, val_labels), (test_dat, test_labels)

def get_mnist_trans(test,shift):
    ll=test.shape[0]
    uu=np.int32(np.random.rand(ll,2)*shift)
    test_t=[]
    for i,t in enumerate(test):
        tt=np.zeros((1,28,28))
        tt[0,0:(28-uu[i,0]),0:(28-uu[i,1])]=t[0,uu[i,0]:,uu[i,1]:]
        test_t.append(tt)
    test_labels=np.int32(np.load(datadir+'mnist/MNIST_labels.npy'))
    test_trans_dat=np.float32(np.concatenate(test_t,axis=0).reshape((-1,1,28,28)))
    print('Transformed test shape',test_trans_dat.shape)
    return (test_trans_dat, test_labels[60000:])

def get_cifar():
    with h5py.File(datadir+'CIFAR/cifar10_train.hdf5', "r") as f:
        tr=f[('data')][:].transpose(0,3,1,2)
        tr_lb=f[('labels')][:]
    tr=np.float32(tr)
    train_data=tr[0:45000]/255.
    train_labels=tr_lb[0:45000]
    val_data=tr[45000:]/255.
    val_labels=tr_lb[45000:]

    with h5py.File(datadir+'CIFAR/cifar10_test.hdf5', "r") as f:
        test_data=f[('data')][:].transpose(0,3,1,2)
        test_data=np.float32(test_data/255.)
        test_labels=f[('labels')][:]
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

def get_data(data_set, train_size=50000):
    if (data_set=="mnist"):
        return(get_mnist(train_size=train_size))
    if (data_set=="cifar"):
        return(get_cifar())

class MNIST_Net(nn.Module):
    def __init__(self,ks=5,p=0.5,minimizer='Adam',device=device):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.drop2 = nn.Dropout2d(p)
        ks=np.int32(ks)
        self.p1=nn.MaxPool2d(kernel_size=[ks],stride=2,padding=[np.int32(ks/2)])
        self.p2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.first=True
        if self.first:
            self.forward(torch.zeros((1,1,28,28)))
        if minimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr = step_size)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr = step_size, momentum=0.9)

        self.criterion=nn.CrossEntropyLoss()

    def forward(self, x):

        x = F.relu(self.p1(self.conv1(x)))
        if self.first:
            print('After conv1',x.shape)
        x = F.relu(self.p2(self.drop2(self.conv2(x))))
        if self.first:
            print('After conv2',x.shape)
            self.first=False
            self.inp=x.shape[1]*x.shape[2]*x.shape[3]
            print('Dimension before first fully connected',self.inp)
            self.fc1 = nn.Linear(self.inp, 256)
            self.fc2 = nn.Linear(256, 10)
        x = x.view(-1, self.inp)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def get_acc_and_loss(self, data, targ):
        output = self.forward(data)
        loss = self.criterion(output, targ)
        pred = torch.max(output,1)[1]
        correct = torch.eq(pred,targ).sum()

        return loss,correct

    def run_grad(self,data,targ):

        loss, correct=self.get_acc_and_loss(data,targ)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, correct

def run_epoch(net,epoch,train,batch_size, num=None, ttype="train",device=device):


    if ttype=='train':
        t1=time.time()
        n=train[0].shape[0]
        if (num is not None):
            n=np.minimum(n,num)
        ii=np.array(np.arange(0,n,1))
        tr=train[0][ii]
        y=train[1][ii]
        train_loss=0; train_correct=0
        with tqdm(total=len(y)) as progress_bar:
            for j in np.arange(0,len(y),batch_size):
                data=torch.torch.from_numpy(tr[j:j+batch_size]).to(device)
                targ=torch.torch.from_numpy(y[j:j+batch_size]).type(torch.long).to(device)
                loss, correct = net.run_grad(data,targ)

                train_loss += loss.item()
                train_correct += correct.item()

                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(data.size(0))
        train_loss /= len(y)
        print('\nTraining set epoch {}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
            train_loss, train_correct, len(y),
            100. * train_correct / len(y)))

def net_test(net,val,batch_size,ttype='val',device=device):
    net.eval()
    with torch.no_grad():
                test_loss = 0
                test_correct = 0
                vald=val[0]
                yval=val[1]
                for j in np.arange(0,len(yval),batch_size):
                    data=torch.from_numpy(vald[j:j+batch_size]).to(device)
                    targ = torch.from_numpy(yval[j:j+batch_size]).type(torch.long).to(device)
                    loss,correct=net.get_acc_and_loss(data,targ)

                    test_loss += loss.item()
                    test_correct += correct.item()

                test_loss /= len(yval)
                SSS='Validation'
                if (ttype=='test'):
                    SSS='Test'
                print('\n{} set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(SSS,
                    test_loss, test_correct, len(yval),
                    100. * test_correct / len(yval)))


def net_test(net,val,batch_size,ttype='val',device=device):
    net.eval()
    with torch.no_grad():
                test_loss = 0
                test_correct = 0
                vald=val[0]
                yval=val[1]
                for j in np.arange(0,len(yval),batch_size):
                    data=torch.from_numpy(vald[j:j+batch_size]).to(device)
                    targ = torch.from_numpy(yval[j:j+batch_size]).type(torch.long).to(device)
                    loss,correct=net.get_acc_and_loss(data,targ)

                    test_loss += loss.item()
                    test_correct += correct.item()

                test_loss /= len(yval)
                SSS='Validation'
                if (ttype=='test'):
                    SSS='Test'
                print('\n{} set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(SSS,
                    test_loss, test_correct, len(yval),
                    100. * test_correct / len(yval)))

import time
batch_size=500
step_size=.001
num_epochs=20
numtrain=50000
minimizer="Adam"
data_set="mnist"
model_name="model"
dropout_p=0.
use_gpu=True

# use GPU when possible
device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'
print(device)
train,val,test=get_data(data_set=data_set, train_size=10000)

net = MNIST_Net(ks=5,p = dropout_p, minimizer=minimizer)
net.to(device)

print('model architecture')
print(net)

print(sum(p.numel() for p in net.parameters()))
print(sum(p.numel() for p in net.parameters() if p.requires_grad))

#define optimizer
train=(train[0][0:numtrain],train[1][0:numtrain])

for i in range(num_epochs):
    run_epoch(net,i,train,batch_size, num=numtrain, ttype="train",device=device)
    net_test(net,val,batch_size,device=device)

print('test on original test data')
net_test(net,test,batch_size,ttype="test")

torch.save(net.state_dict(), predir+model_name)

batch_size=500
step_size=.001
num_epochs=2
minimizer="Adam"
data_set="mnist"
model_name="model"
new_model_name = 'model'
dropout_p =0.
use_gpu=True

# use GPU when possible
device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'

net = MNIST_Net(ks=5,p = dropout_p, minimizer=minimizer)
net.to(device)

train,val,test=get_data(data_set=data_set)

state_dict = torch.load(predir+model_name, map_location = device)
net.load_state_dict(state_dict)

for i in range(num_epochs):
   run_epoch(net,i,train,batch_size, num=numtrain, ttype="train")

torch.save(net.state_dict(), predir+new_model_name)