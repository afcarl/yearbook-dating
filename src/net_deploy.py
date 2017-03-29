import sys
sys.path.insert(0,'../python')
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]) #1, 1, 2, 0
        #param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]) 
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def yearbook_dating(split):
    num_classes = 83
    n = caffe.NetSpec()
    n.data = L.Input(shape=[dict(dim=[1, 3, 96, 96])])
    
    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    # fully connected layers
    # Classification layer and loss
    n.fc6yrbook = L.InnerProduct(n.pool5, num_output=4096, weight_filler=dict(type='xavier'), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]) # 1120
    n.relu6 = L.ReLU(n.fc6yrbook, in_place=True)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7yrbook = L.InnerProduct(n.drop6, num_output=4096, weight_filler=dict(type='xavier'), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.relu7= L.ReLU(n.fc7yrbook, in_place=True)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    n.yrbook = L.InnerProduct(n.drop7, num_output=num_classes, weight_filler=dict(type='gaussian', std=0.005), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.dist = L.Softmax(n.yrbook)
    return n.to_proto()

def make_net():
    with open('deploy.prototxt', 'w') as f:
        f.write(str(yearbook_dating(['deploy'])))


if __name__ == '__main__':
    make_net()
