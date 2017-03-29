import sys
sys.path.insert(0,'../python')
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def conv_bn_scale_relu(bottom, n, name, glob=False, frozen=False, num_output=64, kernel_size=3, stride=1, pad=1):
    if frozen:
        learning_rates = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        learning_rates = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]

    n[name + '_conv'] = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=learning_rates,
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    n[name + '_bn'] = L.BatchNorm(n[name+'_conv'], moving_average_fraction=0.9, use_global_stats=glob, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),  dict(lr_mult=0, decay_mult=0)])
    n[name + '_scale'] = L.Scale(n[name+'_bn'], scale_param=dict(bias_term=True), in_place=True)
    n[name + '_relu'] = L.ReLU(n[name+'_scale'], in_place=True)

    return n[name + '_relu']

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def yearbook_dating(dset, gender, loss, split, bs, frozen=False):
    global_stats = False if 'train' in split else True
    num_classes = 83
    loss = 'softmax' if loss == 0 else 'sigce'
    mean = (121.87, 121.87, 121.87) if gender == 'women' else (137.47, 137.47, 137.47)
    n = caffe.NetSpec()
    pydata_params = dict(loss=loss, im_shape=(96,96), num_classes=num_classes, batch_size = bs, split=split, mean=mean,
            seed=1337)
    pydata_params['yearbook_dir'] = '../data/{}/'.format(dset) + gender
    pylayer = 'YearbookDataLayer'
    n.data, n.label = L.Python(module='yearbook_layers', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

    # the base net
    relu1_1 = conv_bn_scale_relu(n.data, n, '1_1', global_stats, frozen=frozen, num_output=64)
    relu1_2 = conv_bn_scale_relu(relu1_1, n, '1_2', global_stats, frozen=frozen, num_output=64)
    n.pool1 = max_pool(relu1_2)

    relu2_1 = conv_bn_scale_relu(n.pool1, n, '2_1', global_stats, frozen=frozen, num_output=128)
    relu2_2 = conv_bn_scale_relu(relu2_1, n, '2_2', global_stats, frozen=frozen, num_output=128)
    n.pool2 = max_pool(relu2_2)
    
    relu3_1 = conv_bn_scale_relu(n.pool2, n, '3_1', global_stats, frozen=frozen, num_output=256)
    relu3_2 = conv_bn_scale_relu(relu3_1, n, '3_2', global_stats, frozen=frozen, num_output=256)
    relu3_3 = conv_bn_scale_relu(relu3_2, n, '3_3', global_stats, frozen=frozen, num_output=256)
    n.pool3 = max_pool(relu3_3)
    
    relu4_1 = conv_bn_scale_relu(n.pool3, n, '4_1', global_stats, frozen=frozen, num_output=512)
    relu4_2 = conv_bn_scale_relu(relu4_1, n, '4_2', global_stats, frozen=frozen, num_output=512)
    relu4_3 = conv_bn_scale_relu(relu4_2, n, '4_3', global_stats, frozen=frozen, num_output=512)
    n.pool4 = max_pool(relu4_3)
    
    relu5_1 = conv_bn_scale_relu(n.pool4, n, '5_1', global_stats, frozen=frozen, num_output=512)
    relu5_2 = conv_bn_scale_relu(relu5_1, n, '5_2', global_stats, frozen=frozen, num_output=512)
    relu5_3 = conv_bn_scale_relu(relu5_2, n, '5_3', global_stats, frozen=frozen, num_output=512)
    n.pool5 = max_pool(relu5_3)

    # fully connected layers
    n.fc6yrbook = L.InnerProduct(n.pool5, num_output=4096, weight_filler=dict(type='xavier'), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]) # 1120
    n.fc6_bn = L.BatchNorm(n.fc6yrbook, moving_average_fraction=0.9, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),  dict(lr_mult=0, decay_mult=0)]) 
    n.fc6_scale = L.Scale(n.fc6_bn, scale_param=dict(bias_term=True), in_place=True)
    n.relu6 = L.ReLU(n.fc6_scale, in_place=True)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7yrbook = L.InnerProduct(n.drop6, num_output=4096, weight_filler=dict(type='xavier'), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.fc7_bn = L.BatchNorm(n.fc7yrbook, moving_average_fraction=0.9, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),  dict(lr_mult=0, decay_mult=0)]) 
    n.fc7_scale = L.Scale(n.fc7_bn, scale_param=dict(bias_term=True), in_place=True)
    n.relu7= L.ReLU(n.fc7_scale, in_place=True)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    # Classification layer and loss
    n.yrbook = L.InnerProduct(n.drop7, num_output=num_classes, weight_filler=dict(type='gaussian', std=0.005), param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    if loss == 'softmax':
        n.loss = L.SoftmaxWithLoss(n.yrbook, n.label)
    elif loss == 'sigce':
        n.loss = L.SigmoidCrossEntropyLoss(n.yrbook, n.label)

    return n.to_proto()
'''
def make_net():
    with open('train.prototxt', 'w') as f:
        f.write(str(yearbook_dating(['train'], sys.argv[1])))

    with open('val.prototxt', 'w') as f:
        f.write(str(yearbook_dating(['val'], sys.argv[1])))

if __name__ == '__main__':
    make_net()
'''
