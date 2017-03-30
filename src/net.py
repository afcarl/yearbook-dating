import sys
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, freeze, ks=3, stride=1, pad=1):
    if freeze:
        lr_params = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        lr_params = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, param=lr_params)
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def yearbook_dating(gender, split, bs, freeze=False, deploy=False):
    ''' VGG classification architecture
        Fully connected layers are resized for the Yearbook images (96 x 96 pixels)
        Load images into the net via Python data layer '''
    num_classes = 83
    mean = (121.87, 121.87, 121.87) if gender == 'women' else (137.47, 137.47, 137.47)
    im_shape = (96,96)
    n = caffe.NetSpec()
    if not deploy:
        pydata_params = dict(im_shape=im_shape, num_classes=num_classes, batch_size=bs, split=split, mean=mean, seed=1337)
        pydata_params['yearbook_dir'] = 'data/faces/' + gender
        # Specify the data layer to use:
        # - yearbook_layers for loading cropped, aligned portraits
        # - yearbook_bg_layers for loading background crops (used for generalization exp in the paper)
        pylayer = 'YearbookDataLayer'
        n.data, n.label = L.Python(module='yearbook_layers', layer=pylayer,
                ntop=2, param_str=str(pydata_params))
    else:
        n.data = L.Input(shape=[dict(dim=[1, 3, im_shape[0], im_shape[1]])])

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, freeze)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, freeze)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, freeze)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, freeze)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, freeze)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, freeze)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, freeze)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, freeze)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, freeze)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, freeze)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, freeze)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, freeze)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, freeze)
    n.pool5 = max_pool(n.relu5_3)

    # fully connected layers
    n.fc6yrbook = L.InnerProduct(n.pool5, num_output=4096, weight_filler=dict(type='xavier'), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]) 
    n.relu6 = L.ReLU(n.fc6yrbook, in_place=True)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7yrbook = L.InnerProduct(n.drop6, num_output=4096, weight_filler=dict(type='xavier'), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.relu7= L.ReLU(n.fc7yrbook, in_place=True)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    # Classification layer and loss
    n.yrbook = L.InnerProduct(n.drop7, num_output=num_classes, weight_filler=dict(type='gaussian', std=0.005), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    if not deploy:
        n.loss = L.SoftmaxWithLoss(n.yrbook, n.label)
        if split != ['train']:
            n.acc = L.Accuracy(n.yrbook, n.label)

    return n.to_proto()
