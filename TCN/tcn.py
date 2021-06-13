# See https://www.kaggle.com/ceshine/pytorch-temporal-convolutional-networks

# ======================
#     TCN Components
# ======================
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, num_inputs,num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(
            num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        return self.decoder(self.dropout(self.tcn(x)[:, :, -1]))


# ======================
#     Dataloader
# ======================

def create_dl(view_len, batch_size, classes, device):
    """
        Inputs:
            view_len: int, length of the samples; batches are composed of segments classes[idx:idx+view_len] and labels composed of classes[idx+view_len] 
            batch_size: int
            classes: 1-D numpy array of int (classes), the temporal series
            device: torch device
        Outputs:
            dataloader iterating (n, samples) with samples=(batch,label), batch of shape (batch_size, view_len) and label of shape (batch_size)
    """
    def collate_fn(data):
        batch = torch.empty(len(data), view_len, dtype=torch.float32, device=device)
        label = torch.empty(len(data), dtype=torch.long, device=device)
        idx = torch.from_numpy(np.array(data).astype(int)).unsqueeze(1).repeat(1, view_len) + (torch.ones(len(data), view_len, device=device).cumsum(axis=-1)-1)
        idx = idx.view(-1).int().cpu().numpy()
        batch = torch.from_numpy(classes[idx]).view(len(data),-1).float()
        idx = np.array(data).astype(int) + view_len
        label = torch.from_numpy(classes[idx]).long()
        return batch, label
    dl = torch.utils.data.Dataloader((torch.ones(len(classes)-view_len-1).cumsum(axis=0)-1), collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    return dl


# ======================
#     Training
# ======================

def train(dl_train, dl_validation, model, num_epochs):
    """
        Inputs:
            dl_train: dataloader for training set, iterating (n, samples) with samples=(batch,label)
            dl_validation: dataloader for validation set, iterating (n, samples) with samples=(batch,label)
            model: torch.nn.Module to train
            num_epochs: int, number of epochs for training
        Outputs:
            None
        Logging results in a SummaryWriter en './runs'
    """
    writer = SummaryWriter()
    lr = 0.1
    beta = 0.01
    optim = torch.optim.Adam(model.paramters(), lr=lr, betas=(beta, 0.999))
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for n, samples in enumerate(dl_train):
            text = "\rEpoch "+str(epoch+1)+"/"+str(num_epochs)+", training batch "+str(n)+"/"+str(len(dl_train))+"      "
            sys.stdout.write(text)

            batch, labels = samples
            model.zero_grad()
            preds = model(batch.unsqueeze(1))
            loss = loss_fn(preds, labels)
            loss.backward()
            optim.step()
            train_loss += loss.data.item()
            writer.add_scalar('Train Batch Loss', loss.data.item(), epoch*len(dl_train) + n)
        writer.add_scalar('Train Loss', train_loss/(n+1), epoch)

        model.eval()
        val_loss = 0.0
        for n, samples in enumerate(dl_validation):
            text = "\rEpoch "+str(epoch+1)+"/"+str(num_epochs)+", validation batch "+str(n)+"/"+str(len(dl_validation))+"      "
            sys.stdout.write(text)

            batch, labels = samples
            model.zero_grad()
            preds = model(batch.unsqueeze(1))
            loss = loss_fn(preds, labels)
            val_loss += loss.data.item()
            writer.add_scalar('Validation Batch Loss', loss.data.item(), epoch*len(dl_train) + n)
        writer.add_scalar('Validation Loss', val_loss/(n+1), epoch)

        for name, weight in model.named_paramters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f'{name}.grad', weight.grad, epoch)
    
    writer.flush()
    writer.close()
    return None


def main():
    view_len = 1024
    batch_size = 128
    num_epochs = 50
    kernel_size = 5
    dropout = 0.25
    num_channels = [8,12,16]

    # Example of data, put your own here
    nb_classes = 16
    classes = np.random.randint(nb_classes, size=10000)

    idx_split = int(len(classes)*0.8)
    classes_train = classes[:idx_split]
    classes_validation = classes[idx_split:]

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    dl_train = create_dl(view_len, batch_size, classes_train, device)
    dl_validation = create_dl(view_len, batch_size, classes_validation, device)

    model = TCNModel(num_inputs=1, num_channels=num_channels, num_outputs=nb_classes, kernel_size=kernel_size, dropout=dropout)
    model = model.to(device)

    train(dl_train, dl_validation, model, num_epochs)

if __name__ == '__main__':
    main()