import numpy as np
import os
import sys
import net
import caffe
from solver import make_solver
import setproctitle
import time
import random

EXP = sys.argv[1] # descriptive experiment name
setproctitle.setproctitle(EXP) # name the process
GENDER = str(sys.argv[2]) # "men" or "women"
WEIGHTS = str(sys.argv[3]) # path to caffe model weights
GPU = int(sys.argv[4])  # GPU id
LR = float(sys.argv[5]) # SGD learning rate
BATCH = int(sys.argv[6]) # SGD mini-batch size
FROZEN = bool(int(sys.argv[7])) # if true, only train FC layers

# Determinism
np.random.seed(1337)
random.seed(1337)

# Make output directories
expdir = 'output/{}'.format(EXP)
snapshot_path = '{}/snapshots'.format(expdir)
try:
    os.makedirs(expdir)
    os.makedirs(snapshot_path)
except:
    pass

# caffe GPU init
caffe.set_device(GPU) 
caffe.set_mode_gpu()

# Make the net
train_path = '{}/train.prototxt'.format(expdir)
test_path = '{}/val.prototxt'.format(expdir) 
deploy_path = '{}/deploy.prototxt'.format(expdir) 
with open(train_path, 'w') as f:
    f.write(str(net.yearbook_dating(GENDER, ['train'], BATCH, FROZEN)))
with open(test_path, 'w') as f:
    f.write(str(net.yearbook_dating(GENDER, ['test'], 1, FROZEN)))
with open(deploy_path, 'w') as f:
    f.write(str(net.yearbook_dating(GENDER, ['test'], 1, FROZEN, deploy=True)))

# Instantiate the solver
solver_path = make_solver(snapshot_path, train_path, test_path, base_lr=LR, boost=1)
solver = caffe.SGDSolver(solver_path)
testnet = solver.test_nets[0]
testnet.share_with(solver.net) # tie the weights of test net with train net

# copy base weights for fine-tuning
solver.net.copy_from(WEIGHTS)

# Check that we copied the params by printing a summary
def print_net_params(net):
    for k,v in net.params.iteritems():
        print k, np.mean(np.abs(v[0].data))
print_net_params(solver.net)

# Training Loop
niter = 100
num_steps = 200
acc_accum = np.zeros(niter)
train_loss = np.zeros(niter)
num_test_examples = len(open('data/faces/{}/test.txt'.format(GENDER)).read().splitlines())
for it in range(niter):
    print 'Iteration', it
    solver.step(num_steps) # roughly one epoch with batch size 64
    loss = solver.net.blobs['loss'].data
    train_loss[it] = loss
    print 'Loss:', train_loss[it]
    correct = 0
    for _ in range(num_test_examples): 
        testnet.forward()
        correct += sum(testnet.blobs['yrbook'].data.argmax(1) == np.squeeze(testnet.blobs['label'].data).astype(np.int32))
    accuracy = correct/float(num_test_examples)
    acc_accum[it] = accuracy
    np.save('{}/train_loss.npy'.format(expdir), train_loss)
    np.save('{}/accuracy.npy'.format(expdir), acc_accum)
    print 'Accuracy at iter', str((it+1)*num_steps), ':', accuracy
    if it % 10 == 0 and it > 0:
        solver.net.save('{}/snapshots/train_iter_{}.caffemodel'.format(expdir, str(num_steps*it)))
    print_net_params(solver.net)

