import numpy as np
import os
import sys
import net
import alexnet
import net_bn
import caffe
from solver import make_solver
import setproctitle
import time
import random

EXP = sys.argv[1]
setproctitle.setproctitle(EXP)
GENDER = str(sys.argv[2])
WEIGHTS=str(sys.argv[3])
GPU = int(sys.argv[4]) 
LR = float(sys.argv[5])
BATCH = int(sys.argv[6])
SEED = int(sys.argv[7])
FROZEN = bool(int(sys.argv[8])) # if true, only train FC layers

# Determinism
np.random.seed(SEED)
random.seed(SEED)

expdir = '../output/{}'.format(EXP)
snapshot_path = '{}/snapshots'.format(expdir)
try:
    os.makedirs(expdir)
    os.makedirs(snapshot_path)
except:
    pass

# init
caffe.set_device(GPU) 
caffe.set_mode_gpu()

# Make the net
# TODO: no batch norm
train_path = '{}/train.prototxt'.format(expdir)
test_path = '{}/val.prototxt'.format(expdir) 
with open(train_path, 'w') as f:
    f.write(str(net_bn.yearbook_dating(DSET, GENDER, SIGCE, ['train'], BATCH, FROZEN)))
    #f.write(str(net.yearbook_dating(GENDER, SIGCE, ['train'], BATCH, FROZEN)))
with open(test_path, 'w') as f:
    f.write(str(net_bn.yearbook_dating(DSET, GENDER, SIGCE, ['test'], 1, FROZEN)))
    #f.write(str(net.yearbook_dating(GENDER, SIGCE, ['test'], 1, FROZEN)))

# Instantiate the solver
solver_path = make_solver(snapshot_path, train_path, test_path, base_lr=LR, boost=1)
solver = caffe.SGDSolver(solver_path)

# copy base weights for fine-tuning
solver.net.copy_from(WEIGHTS)

# Check that we copied the params we wanted
def print_net_params(net):
    for k,v in net.params.iteritems():
        print k, np.mean(np.abs(v[0].data))
print_net_params(solver.net)

# run for 100k iterations
niter = 200
acc_accum = np.zeros(niter)
train_loss = np.zeros(niter)
testnet = solver.test_nets[0]
num_test_examples = len(open('../data/faces/{}/test.txt'.format(GENDER)).read().splitlines())
for it in range(niter):
    print 'Iteration', it
    solver.step(200) # roughly one epoch with batch size 64
    loss = solver.net.blobs['loss'].data
    train_loss[it] = loss
    print 'Loss:', train_loss[it]
    testnet.share_with(solver.net) # share the weights from train net with test net
    correct = 0
    for _ in range(num_test_examples): 
        testnet.forward()
        correct += sum(testnet.blobs['yrbook'].data.argmax(1) == np.squeeze(testnet.blobs['label'].data).astype(np.int32))
    accuracy = correct/float(num_test_examples)
    acc_accum[it] = accuracy
    np.save('{}/train_loss.npy'.format(expdir), train_loss)
    np.save('{}/accuracy.npy'.format(expdir), acc_accum)
    print 'Accuracy at iter', str((it+1)*1000), ':', accuracy
    if it % 10 == 0 and it > 0:
        solver.net.save('{}/snapshots/train_iter_{}.caffemodel'.format(expdir, str(400*it)))
    print_net_params(solver.net)

