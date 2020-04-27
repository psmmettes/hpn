#
# Obtain hyperspherical prototypes prior to network training.
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
import random
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from   torch import nn

#
# PArse user arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperspherical prototypes")
    parser.add_argument('-c', dest="classes", default=100, type=int)
    parser.add_argument('-d', dest="dims", default=100, type=int)
    parser.add_argument('-l', dest="learning_rate", default=0.1, type=float)
    parser.add_argument('-m', dest="momentum", default=0.9, type=float)
    parser.add_argument('-e', dest="epochs", default=10000, type=int,)
    parser.add_argument('-s', dest="seed", default=300, type=int)
    parser.add_argument('-r', dest="resdir", default="", type=str)
    parser.add_argument('-w', dest="wtvfile", default="", type=str)
    parser.add_argument('-n', dest="nn", default=2, type=int)
    args = parser.parse_args()
    return args

#
# Compute the loss related to the prototypes.
#
def prototype_loss(prototypes):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototypes, prototypes.t()) + 1
    # Remove diagnonal from loss.
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()

#
# Compute the semantic relation loss.
#
def prototype_loss_sem(prototypes, triplets):
    product = torch.matmul(prototypes, prototypes.t()) + 1
    product -= 2. * torch.diag(torch.diag(product))
    loss1 = -product[triplets[:,0], triplets[:,1]]
    loss2 = product[triplets[:,2], triplets[:,3]]
    return loss1.mean() + loss2.mean(), product.max()

#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user arguments.
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    kwargs = {'num_workers': 64, 'pin_memory': True}

    # Set seed.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize prototypes and optimizer.
    if os.path.exists(args.wtvfile):
        use_wtv = True
        wtvv = np.load(args.wtvfile)
        for i in xrange(wtvv.shape[0]):
            wtvv[i] /= np.linalg.norm(wtvv[i])
        wtvv = torch.from_numpy(wtvvectors)
        wtvsim = torch.matmul(wtvv, wtvv.t()).float()
        
        # Precompute triplets.
        nns, others = [], []
        for i in xrange(wtvv.shape[0]):
            sorder = np.argsort(wtvsim[i,:])[::-1]
            nns.append(sorder[:args.nn])
            others.append(sorder[args.nn:-1])
        triplets = []
        for i in xrange(wtvv.shape[0]):
            for j in xrange(len(nns[i])):
                for k in xrange(len(others[i])):
                    triplets.append([i,j,i,k])
        triplets = np.array(triplets).astype(int)
    else:
        use_wtv = False
    
    # Initialize prototypes.
    prototypes = torch.randn(args.classes, args.dims)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
    optimizer = optim.SGD([prototypes], lr=args.learning_rate, \
            momentum=args.momentum)

    # Optimize for separation.
    for i in xrange(args.epochs):
        # Compute loss.
        loss1, sep = prototype_loss(prototypes)
        if use_wtv:
            loss2 = prototype_loss_sem(prototypes, triplets)
            loss = loss1 + loss2
        else:
            loss = loss1
        # Update.
        loss.backward()
        optimizer.step()
        # Renormalize prototypes.
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
        optimizer = optim.SGD([prototypes], lr=args.learning_rate, \
                momentum=args.momentum)
        print "%03d/%d: %.4f\r" %(i, args.epochs, sep),
        sys.stdout.flush()
    print
    
    # Store result.
    np.save(args.resdir + "prototypes-%dd-%dc.npy" %(args.dims, args.classes), \
            prototypes.data.numpy())
