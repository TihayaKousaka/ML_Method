# 🎈 Usage 

You need to install the requirements:
This has been tested with Python `v3.6.8`, Torch `v1.3.1` and Torchvision `v0.4.2`.

```shell
pip install -r requirements.txt
```

The are various arguments available for training and testing the network in addition to showing example data. When training or testing with uncertainty provide the `--uncertainty` argument in addition to one of the following for loss: `--mse`, `--digamma`, `--log`.

```
python main.py --help

usage: main.py [-h] [--train] [--epochs EPOCHS] [--dropout] [--uncertainty]
               [--mse] [--digamma] [--log] [--test] [--examples]

optional arguments:
  -h, --help       show this help message and exit
  --train          To train the network.
  --epochs EPOCHS  Desired number of epochs.
  --dropout        Whether to use dropout or not.
  --uncertainty    Use uncertainty or not.
  --mse            Set this argument when using uncertainty. Sets loss
                   function to Expected Mean Square Error.
  --digamma        Set this argument when using uncertainty. Sets loss
                   function to Expected Cross Entropy.
  --log            Set this argument when using uncertainty. Sets loss
                   function to Negative Log of the Expected Likelihood.
  --test           To test the network.
  --examples       To example MNIST data.
```

Example of how to train the network:

```shell
python main.py --train --dropout --uncertainty --mse --epochs 50
```


Example of how to test the network:

```shell
python main.py --test --uncertainty --mse
```