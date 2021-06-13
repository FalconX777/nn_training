# Neural networks training pipelines (PyTorch)

Functions with optimized dataloaders for training NN (GAN and TCN) on a unique temporal series (code adaptable to several ones). 

## GAN

Vanilla GAN architecture to sample float temporal series.  

Run "python gan.py" to see an example with a random dataset.

## TCN

TCN to predict the next class of a int (class) temporal series, knowing the (view_len)-th previous ones.  

Run "python tcn.py" to see an example with a random dataset.