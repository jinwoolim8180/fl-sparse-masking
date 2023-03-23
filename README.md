# Federated Learning
An open source ferderated learning implement based on Pytorch.

Dataset: MNIST, Cifar-10, FEMNIST, Fashion-MNIST, Shakespeare.

## Dataset
Your need to download dataset, except for Femnist and Shakespeare dataset, since this repository already includes the dataset.

You can run the code and other datasets will be downloaded automatically.

### Mnist
IID:

python main.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0 --lr 0.01

non-IID:

python main.py --dataset mnist --num_channels 1 --model cnn --epochs 50 --gpu 0 --lr 0.01
### Femnist
Femnist is naturally non-IID

This dataset is sampled from project leaf: https://leaf.cmu.edu/, using command 

./preprocess.sh -s niid --sf 0.05 -k 0 -t sample (small-sized dataset)

You can run using this:

python main.py --dataset femnist --num_channels 1 --model cnn --epochs 50 --gpu 0 --lr 0.01

### Shakespeare

Shakespeare is naturally non-IID

This dataset is sampled from project leaf: https://leaf.cmu.edu/, using command 

./preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8

You can run using this:

python main.py --dataset shakespeare --model lstm --epochs 50 --gpu 0 --lr 1.4

### Cifar-10
IID:

python main.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0 --lr 0.02

non-IID:

python main.py --dataset cifar --num_channels 3 --model cnn --epochs 50 --gpu 0 --lr 0.02

### Fashion-Mnist

IID:

python main.py --dataset fashion-mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0 --lr 0.01 

non-IID:

python main.py --dataset fashion-mnist --num_channels 1 --model cnn --epochs 50 --gpu 0 --lr 0.01


