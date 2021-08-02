import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Generator(torch.nn.Module):
    def __init__(self, dim_in, dim_med, dim_out):
        super(Generator, self).__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_med),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_med, dim_out)
        )
        self.dim_noise = dim_in
    def __call__(self, z):
        return self.nn(z)

class Discriminator(torch.nn.Module):
    def __init__(self, dim_in, dim_med):
        super(Generator, self).__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_med),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_med, 1),
            torch.nn.Sigmoid()
        )
        self.dim_noise = dim_in
    def __call__(self, z):
        return self.nn(z)


# ======================
#     Dataloader
# ======================

def create_dl(view_len, batch_size, series, device):
    """
        Inputs:
            view_len: int, length of the samples; batches are composed of segments series[idx:idx+view_len]
            batch_size: int
            series: 1-D numpy array of float (series), the temporal series
            device: torch device
        Outputs:
            dataloader iterating (n, batch), batch of shape (batch_size, view_len)
    """
    def collate_fn(data):
        batch = torch.empty(len(data), view_len, dtype=torch.float32, device=device)
        idx = torch.from_numpy(np.array(data).astype(int)).unsqueeze(1).repeat(1, view_len) + (torch.ones(len(data), view_len, device=device).cumsum(axis=-1)-1)
        idx = idx.view(-1).int().cpu().numpy()
        batch = torch.from_numpy(series[idx]).view(len(data),-1).float()
        return batch
    dl = torch.utils.data.Dataloader((torch.ones(len(series)-view_len-1).cumsum(axis=0)-1), collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    return dl


# ======================
#     Training
# ======================

def train(dl_train, dl_validation, disc, gen, num_epochs, nb_disc, nb_gen):
    """
        Inputs:
            dl_train: dataloader for training set, iterating (n, batch), batch of shape (batch_size, view_len)
            dl_validation: dataloader for validation set, iterating (n, batch), batch of shape (batch_size, view_len)
            model: torch.nn.Module to train
            num_epochs: int, number of epochs for training
        Outputs:
            None
        Logging results in a SummaryWriter en './runs'
    """
    writer = SummaryWriter()
    lr = 0.1
    beta = 0.01
    optim_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta, 0.999))
    optim_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta, 0.999))
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        acc_loss_disc = 0.0
        acc_loss_gen = 0.0
        for n, batch in enumerate(dl_train):
            text = "\rEpoch "+str(epoch+1)+"/"+str(num_epochs)+", training batch "+str(n)+"/"+str(len(dl_train))+"      "
            sys.stdout.write(text)

            # disc step
            for _ in range(nb_disc):
                disc.zero_grad()
                noise = torch.empty((len(batch), gen.dim_noise)).normal_()
                D_scores_on_real = disc(batch)
                D_scores_on_fake = disc(gen(noise))
                loss_d = -torch.mean(torch.log(1-D_scores_on_fake) + torch.log(D_scores_on_real))
                loss_d.backward()
                optim_disc.step()
            acc_loss_disc += loss_d.data.item()

            # gen step
            for _ in range(nb_gen):
                gen.zero_grad()
                noise = torch.empty((len(batch), gen.dim_noise)).normal_()
                D_scores_on_fake = disc(gen(noise))
                loss_g = -torch.mean(torch.log(1-D_scores_on_fake))
                loss_g.backward()
                optim_gen.step()
            acc_loss_gen += loss_g.data.item()

        writer.add_scalar('Train Discriminator Loss', acc_loss_disc/(n+1), epoch)
        writer.add_scalar('Train Generator Loss', acc_loss_gen/(n+1), epoch)

        model.eval()
        acc_loss_disc = 0.0
        acc_loss_gen = 0.0
        for n, batch in enumerate(dl_validation):
            text = "\rEpoch "+str(epoch+1)+"/"+str(num_epochs)+", validation batch "+str(n)+"/"+str(len(dl_validation))+"      "
            sys.stdout.write(text)

            # disc step
            for _ in range(nb_disc):
                disc.zero_grad()
                noise = torch.empty((len(batch), gen.dim_noise)).normal_()
                D_scores_on_real = disc(batch)
                D_scores_on_fake = disc(gen(noise))
                loss_d = -torch.mean(torch.log(1-D_scores_on_fake) + torch.log(D_scores_on_real))
            acc_loss_disc += loss_d.data.item()

            # gen step
            for _ in range(nb_gen):
                gen.zero_grad()
                noise = torch.empty((len(batch), gen.dim_noise)).normal_()
                D_scores_on_fake = disc(gen(noise))
                loss_g = -torch.mean(torch.log(1-D_scores_on_fake))
            acc_loss_gen += loss_g.data.item()

        writer.add_scalar('Validation Discriminator Loss', acc_loss_disc/(n+1), epoch)
        writer.add_scalar('Validation Generator Loss', acc_loss_gen/(n+1), epoch)

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

    dim_in_g = 1024
    dim_med_g = 512
    dim_out_g = view_len

    dim_in_d = view_len
    dim_med_d = 512

    nb_disc = 10
    nb_gen = 1

    # Example of data, put your own here
    series = np.random.random(size=10000).cumsum(axis=0)

    idx_split = int(len(series)*0.8)
    series_train = series[:idx_split]
    series_validation = series[idx_split:]

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    dl_train = create_dl(view_len, batch_size, series_train, device)
    dl_validation = create_dl(view_len, batch_size, series_validation, device)

    gen = Generator(dim_in_g, dim_med_g, dim_out_g)
    gen = gen.to(device)
    disc = Discriminator(dim_in_d, dim_med_d)
    disc = disc.to(device)

    train(dl_train, dl_validation, disc, gen, num_epochs, nb_disc, nb_gen)

if __name__ == '__main__':
    main()
