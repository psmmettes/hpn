#
# Perform classification on CIFAR-100 using Hypershperical
# Prototype Networks.
#
# @inproceedings{mettes2016hyperspherical,
#  title={Hyperspherical Prototype Networks},
#  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
#  booktitle={Advances in Neural Information Processing Systems},
#  year={2019}
# }
#

import os
import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torchvision import datasets, transforms

import helper
sys.path.append("models/cifar/")
import densenet, resnet, convnet
    

################################################################################
# Training epoch.
################################################################################

#
# Main training function.
#
# model (object)      - Network.
# device (torch)      - Torch device, e.g. CUDA or CPU.
# trainloader (torch) - Training data.
# optimizer (torch)   - Type of optimizer.
# f_loss (torch)      - Loss function.
# epoch (int)         - Epoch iteration.
#
def main_train(model, device, trainloader, optimizer, f_loss, epoch):
    # Set mode to training.
    model.train()
    avgloss, avglosscount = 0., 0.
    
    # Go over all batches.
    for bidx, (data, target) in enumerate(trainloader):
        # Data to device.
        nlabels = target.clone()
        target = model.polars[target]
        data   = torch.autograd.Variable(data).cuda()
        target = torch.autograd.Variable(target).cuda()
        
        # Compute outputs and losses.
        output = model(data)
        loss = (1 - f_loss(output, target)).pow(2).sum()
        
        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update loss.
        avgloss += loss.item()
        avglosscount += 1.
        newloss = avgloss / avglosscount
        
        # Print updates.
        print "Training epoch %d: loss %8.4f - %.0f\r" \
                %(epoch, newloss, 100.*(bidx+1)/len(trainloader)),
        sys.stdout.flush()
    print

################################################################################
# Testing epoch.
################################################################################

#
# Main test function.
#
# model (object)             - Network.
# device (torch)             - Torch device, e.g. CUDA or CPU.
# testloader (torch)         - Test data.
#
def main_test(model, device, testloader):
    # Set model to evaluation and initialize accuracy and cosine similarity.
    model.eval()
    cos = nn.CosineSimilarity(eps=1e-9)
    acc = 0
    
    # Go over all batches.
    with torch.no_grad():
        for data, target in testloader:
            # Data to device.
            data = torch.autograd.Variable(data).cuda()
            target = target.cuda(async=True)
            target = torch.autograd.Variable(target)
            
            # Forward.
            output = model(data).float()
            output = model.predict(output).float()
                
            pred = output.max(1, keepdim=True)[1]
            acc += pred.eq(target.view_as(pred)).sum().item()
    
    # Print results.
    testlen = len(testloader.dataset)
    print "Testing: classification accuracy: %d/%d - %.3f" \
            %(acc, testlen, 100. * acc / testlen)
    return acc / float(testlen)

################################################################################
# Main entry point of the script.
################################################################################

#
# Parse all user arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-100 classification")
    parser.add_argument("--datadir", dest="datadir", default="dat/", type=str)
    parser.add_argument("--resdir", dest="resdir", default="res/", type=str)
    parser.add_argument("--hpnfile", dest="hpnfile", default="", type=str)

    parser.add_argument("-n", dest="network", default="resnet32", type=str)
    parser.add_argument("-r", dest="optimizer", default="sgd", type=str)
    parser.add_argument("-l", dest="learning_rate", default=0.01, type=float)
    parser.add_argument("-m", dest="momentum", default=0.9, type=float)
    parser.add_argument("-c", dest="decay", default=0.0001, type=float)
    parser.add_argument("-s", dest="batch_size", default=128, type=int)
    parser.add_argument("-e", dest="epochs", default=250, type=int)
    parser.add_argument("--seed", dest="seed", default=100, type=int)
    parser.add_argument("--drop1", dest="drop1", default=100, type=int)
    parser.add_argument("--drop2", dest="drop2", default=200, type=int)
    args = parser.parse_args()
    return args

#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user parameters and set device.
    args     = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device   = torch.device("cuda")
    kwargs   = {'num_workers': 32, 'pin_memory': True}

    # Set the random seeds.
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Load data.
    batch_size = args.batch_size
    trainloader, testloader = helper.load_cifar100(args.datadir, \
            batch_size, kwargs)
    nr_classes = 100

    # Load the polars and update the trainy labels.
    classpolars = torch.from_numpy(np.load(args.hpnfile)).float()
    args.output_dims = int(args.hpnfile.split("/")[-1].split("-")[1][:-1])
    
    # Load the model.
    if args.network == "resnet32":
        model = resnet.ResNet(32, args.output_dims, 1, classpolars)
    elif args.network == "densenet121":
        model = densenet.DenseNet121(args.output_dims, classpolars)
    model = model.to(device)
    
    # Load the optimizer.
    optimizer = helper.get_optimizer(args.optimizer, model.parameters(), \
            args.learning_rate, args.momentum, args.decay)
    
    # Initialize the loss functions.
    f_loss = nn.CosineSimilarity(eps=1e-9).cuda()

    # Main loop.
    testscores = []
    learning_rate = args.learning_rate
    for i in xrange(args.epochs):
        print "---"
        # Learning rate decay.
        if i in [args.drop1, args.drop2]:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        # Train and test.
        main_train(model, device, trainloader, optimizer, f_loss, i)
        if i % 10 == 0 or i == args.epochs - 1:
            t = main_test(model, device, testloader)
            testscores.append([i,t])
