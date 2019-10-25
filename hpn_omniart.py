#
# Perform joint classification and regresison on OmniArt using Hypershperical
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
import argparse
import pandas as pd
import numpy as np
from   PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import transforms, datasets
import torchvision.models as models

import helper
sys.path.append("models/omniart/")
import resnet

################################################################################
# Load the OmniArt dataset.
################################################################################

#
# Image loading.
#
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

#
# Image loading.
#
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

#
# Image loading.
#
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

#
# Custom dataset for the OmniArt dataset.
#
class OmniArtDataset(torch.utils.data.Dataset):
    
    #
    # Initialize the dataset with appropriate folders and transforms.
    #
    def __init__(self, root, datafile, transforms, c1=None, c2=None):
        self.root      = root
        self.transform = transforms
        
        # Images and regression data.
        self.data   = pd.read_csv(datafile)
        self.images = self.data["omni_id"].iloc[:]
        self.years  = np.array(self.data["creation_year"].iloc[:])
        self.len    = len(self.years)
        
        # Artist and style data (centuries not used in evaluation).
        self.centuries = np.array(self.data["century"].iloc[:])
        self.styles = np.array(self.data["school"].iloc[:])
        if c1 is None:
            c1 = np.unique(self.centuries)
            self.c1 = c1
        if c2 is None:
            c2 = np.unique(self.styles)
            self.c2 = c2
        
        # Assign names to class ids.
        self.centurylabels = np.zeros(self.len, dtype=int) - 1
        self.stylelabels = np.zeros(self.len, dtype=int) - 1
        self.toremove = []
        for i in xrange(self.len):
            aidx = np.where(self.centuries[i] == c1)[0]
            if len(aidx) == 1:
                self.centurylabels[i] = aidx[0]
            sidx = np.where(self.styles[i] == c2)[0]
            if len(sidx) == 1:
                self.stylelabels[i] = sidx[0]

    #
    # Get an example with the labels.
    #
    def __getitem__(self, index):
        image = self.root + str(self.images[index]) + ".jpg"
        image = pil_loader(image)
        image = self.transform(image)
        year  = self.years[index]
        century = self.centurylabels[index]
        school = self.stylelabels[index]
        return (image, year, school)
    
    #
    # Dataset size.
    #
    def __len__(self):
        return self.len

#
# Load the complete dataset with classification and regression labels.
#
def load_omniart(basedir, trainfile, testfile, batch_size, kwargs):
    # Transformations.
    mrgb = [0.485, 0.456, 0.406]
    srgb = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mrgb, std=srgb)
    ])
    
    # Train set.
    trainset = OmniArtDataset(basedir+"train/", trainfile, transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, \
            shuffle=True, **kwargs)
    
    # Test set.
    c1, c2 = trainset.c1, trainset.c2
    testset    = OmniArtDataset(basedir+"test/", testfile, transform, c1, c2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, \
            shuffle=True, **kwargs)

    return trainloader, testloader

################################################################################
# Training and testing.
################################################################################

#
# Main training function.
#
# model (object)        - Network.
# device (torch)        - Torch device, e.g. CUDA or CPU.
# trainloader (torch)   - Training data.
# optimizer (torch)     - Type of optimizer.
# f_loss                - Loss function.
# allpolars             - Multi-task prototypes.
# epoch (int)           - Epoch iteration.
# task (int)            - Individual or joint ask.
# weight (float)        - Optional weighting between tasks.
#
def main_train(model, device, trainloader, optimizer, f_loss, allpolars, \
        epoch, task, weight=0.5):
    # Set mode to training and initialize the cosine similarity.
    model.train()
    avgloss, avglosscount = 0., 0.
    classloss = nn.CrossEntropyLoss().cuda()
    regloss = nn.MSELoss().cuda()
    
    # Go over all batches.
    for bidx, (data, target1, target2) in enumerate(trainloader):
        # Data to device.
        data    = torch.autograd.Variable(data).cuda()
        target1 = target1.cuda(async=True).float()
        target1 = torch.autograd.Variable(target1)
        target3 = target2.clone().cuda()
        target2 = allpolars[1][target2]
        target2 = target2.cuda(async=True).float()
        target2 = torch.autograd.Variable(target2)
        
        output = model(data)
        # Regression loss.
        upp = allpolars[0].view(-1,1).repeat(1, output.shape[0]).to(device).t()
        loss1 = (target1 - f_loss(output, upp)).pow(2).sum()
        # Classification loss
        loss2 = (1 - f_loss(output[:,1:], target2)).pow(2).sum()
        
        if task == 0:
            loss = loss1
        elif task == 1:
            loss = loss2
        elif task == 2:
            loss = (1.-weight) * loss1 + weight * loss2
        
        # Backpropagation.  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avgloss += loss.item()
        avglosscount += 1.
        newloss = avgloss / avglosscount
        
        # Optionally, print updates.
        print "Training epoch %d: loss %8.4f - %.0f\r" \
                %(epoch, newloss, 100.*(bidx+1)/len(trainloader)),
        sys.stdout.flush()
    print

#
# Main test function.
#
# model (object)        - Network.
# device (torch)        - Torch device, e.g. CUDA or CPU.
# testloader (torch)    - Test data.
# allpolars             - Multi-task prototypes.
# lb (float)            - Lower regression bound.
# ub (Float)            - Upper regression bound.
# task (int)            - Individual or joint ask.
#
def main_test(model, device, testloader, allpolars, lb, ub, task):
    # Set model to evaluation and initialize accuracy and cosine similarity.
    model.eval()
    cos  = nn.CosineSimilarity(eps=1e-9)
    
    mae, acc = 0., 0.
    testlen = len(testloader.dataset)
    pes = np.zeros((testlen, 4))
    pidx = 0
    ccp = allpolars[1].t().cuda()
    
    # Go over all batches.
    with torch.no_grad():
        for bidx, (data, target1, target2) in enumerate(testloader):
            # Data to device.
            data    = torch.autograd.Variable(data).cuda()
            target1 = target1.cuda(async=True).float()
            target1 = torch.autograd.Variable(target1)
            target2 = target2.cuda(async=True)
            target2 = torch.autograd.Variable(target2)

            # Compute outputs and matches.
            output = model(data)
            pred, pred2 = -1, -1
            # Match outputs to upper polar.
            if task == 0 or task == 2:
                upp = allpolars[0].view(-1,1).repeat(1, output.shape[0]).to(device).t()
                scores = (cos(output, upp) + 1) / 2.
                pred = (scores * (ub - lb)) + lb
                mae += torch.abs(pred - target1).sum()
            if task == 1 or task == 2:
                # L2 norm, dot product.
                output2 = F.normalize(output[:,1:], p=2, dim=1)
                output2 = torch.mm(output2, ccp)
                pred2 = output2.max(1, keepdim=True)[1]
                acc += pred2.eq(target2.view_as(pred2)).sum().item()
                pred2 = pred2[:,0]

            pes[pidx:pidx+data.shape[0],0] = target1
            pes[pidx:pidx+data.shape[0],1] = pred
            pes[pidx:pidx+data.shape[0],2] = target2
            pes[pidx:pidx+data.shape[0],3] = pred2
            pidx += data.shape[0]
    
    print "MAE: %.4f -- ACC: %.4f" %(mae / testlen, 100. * acc / testlen)
    return mae / float(testlen), acc / float(testlen), pes
            

################################################################################
# Main entry point of the script.
################################################################################

#
# Parse all user arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Polar Prototypical Regression")
    parser.add_argument("--basedir", dest="basedir", default="dat/omniart/", type=str)
    parser.add_argument("--resdir", dest="resdir", default="res/joint/", type=str)
    parser.add_argument("--multigpu", dest="multigpu", default=0, type=int)
    parser.add_argument("-n", dest="network", default="resnet32", type=str)
    parser.add_argument("-o", dest="output_dims", default=47, type=int)
    parser.add_argument("-r", dest="optimizer", default="sgd", type=str)
    parser.add_argument("-l", dest="learning_rate", default=0.001, type=float)
    parser.add_argument("-m", dest="momentum", default=0.9, type=float)
    parser.add_argument("-c", dest="decay", default=0.0001, type=float)
    parser.add_argument("-s", dest="batch_size", default=64, type=int)
    parser.add_argument("-e", dest="epochs", default=250, type=int)
    parser.add_argument("--seed", dest="seed", default=100, type=int)
    parser.add_argument("--drop1", dest="drop1", default=100, type=int)
    parser.add_argument("--drop2", dest="drop2", default=200, type=int)
    parser.add_argument("--task", dest="task", default=2, type=int)
    parser.add_argument("--weight", dest="weight", default=0.5, type=float)
    args = parser.parse_args()
    return args

#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user parameters and set device.
    args   = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    kwargs = {'num_workers': 64, 'pin_memory': True}
    
    # Set the random seeds.
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
   
    # Load data.
    batch_size = args.batch_size
    root       = args.basedir
    trainfile  = root + "train_complete.csv"
    testfile   = root + "test_complete.csv"
    trainloader, testloader = load_omniart(root, trainfile, testfile, \
            batch_size, kwargs)
   
    # Regression polars and bounds.
    years = trainloader.dataset.years
    lb, ub = 1000., 1932.
    # Initialize the polar bound polar.
    yearpolars = torch.zeros(args.output_dims)
    yearpolars[0] = 1
    # Determine the polar regression values.
    trainy = 2. * ((years - lb) / (ub - lb)) - 1
    trainloader.dataset.years = trainy
    trainloader.dataset.oldyears = years.copy()
    
    # Other prototype loading.
    stylepolars = np.load("dat/prototypes/sgd/prototypes-46d-46c.npy")
    stylepolars = torch.from_numpy(stylepolars).float()
    allpolars = [yearpolars, stylepolars]
    
    # Network type.
    if args.network == "std":
        model = convnet.Std(args.output_dims, None)
    elif args.network == "resnet16":
        model = resnet.ResNet18(args.output_dims, None)
    elif args.network == "resnet32":
        model = resnet.ResNet34(args.output_dims, None)
    model = model.to(device)
    
    # To CUDA.
    if args.multigpu == 1:
        model = torch.nn.DataParallel(model.cuda())
    else:
        model = model.to(device)

    # Network parameters.
    optimname = args.optimizer
    lr        = args.learning_rate
    momentum  = args.momentum
    decay     = args.decay
    params    = model.parameters()
    # Set the optimizer.
    optimizer = helper.get_optimizer(optimname, params, lr, momentum, decay)
    
    # Initialize the loss functions.
    f_loss = nn.CosineSimilarity(eps=1e-9).cuda()

    resdir = args.resdir + "omniart/"
    args.do_norm  = 1
    testscores = []

    # Iterative optimization.
    for i in xrange(args.epochs):
        print "---"
        # Update learning rate.
        if i in [args.drop1, args.drop2]:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Train and test.
        main_train(model, device, trainloader, optimizer, f_loss, allpolars, \
                i, args.task, args.weight)
        if i % 10 == 0 or i == args.epochs - 1:
            s1, s2, pes = main_test(model, device, testloader, allpolars, lb, \
                    ub, args.task)
            testscores.append([i,s1,s2])
