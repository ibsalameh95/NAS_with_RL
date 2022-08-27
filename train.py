import argparse
from pickle import FALSE

import torch
import torchvision
import torchvision.transforms as transforms

from policy_gradient import PolicyGradient


def parse_args():
    desc = "PyTroch implementation of 'Neural Architecture Search with Reinforcement Learning'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--use_cuda', default=False, help='use GPU')
    parser.add_argument('--mode', default='generate')
    parser.add_argument('--dataset', default='MNIST')

    args = parser.parse_args()
    return args

class Params:
    NUM_EPOCHS = 5      # For how many epoches you want to train the selected architichture
    ALPHA = 5e-3        # Learning rate
    BATCH_SIZE = 25     # How many episodes we want to pack into an epoch
    HIDDEN_SIZE = 64    # Number of hidden nodes we have in our cnn
    BETA = 0.1          # The entropy bonus multiplier
    INPUT_SIZE = 3      # Initial input for RNN
    ACTION_SPACE = 3    # Limits for hyperparameters
    NUM_STEPS = 6       # steps for RNN

def main():
    args = parse_args()
    use_cuda = args.use_cuda
    in_channels = 0
    if args.dataset == 'MNIST': 
        in_channels = 1
        dataset='MNIST'
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])  
                    


        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    if args.dataset == 'CIFAR10':
        in_channels = 3
        dataset='CIFAR10'
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=2)

   
    if args.mode == 'generate': 
       policy_gradient = PolicyGradient(config=Params, train_set=trainloader, test_set=testloader, use_cuda=use_cuda, in_channels=in_channels, dataset=dataset)
       policy_gradient.solve_environment()

    if args.mode == 'evaluate': 
       policy_gradient = PolicyGradient(config=Params, train_set=trainloader, test_set=testloader, use_cuda=use_cuda, in_channels=in_channels, dataset=dataset)
       policy_gradient.solve_environment()
    
if __name__ == "__main__":
    main()