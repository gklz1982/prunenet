# merge conv1 into the network and get *training*, *test* accuracy.
import os
os.environ['SKIP_TRAINING_DATA'] = ' ' # skip training data.
from common import *
import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array


name = os.environ.get('name')
if not name:
    name = 'stride'

layer = os.environ.get('layer')
if not layer:
    layer = 'conv1'

print '[environ] name = ', name
print '[environ] layer = ', layer


batch_size = 32
RESULT_NAME = '%s-%s.model' % (name, layer)

im_input = np.zeros((batch_size, 3, th, tw), dtype=np.float32)

# load original model.
net = caffe.Net(MODEL_FILE, WEIGHTS_FILE)
net.set_mode_gpu()
net.blobs['data'].reshape(*im_input.shape)

acc = 0
total = 0

# test accuracy.
dataset = test_set
labels = [0] * len(dataset)

# load prunenet filter.
with open(RESULT_NAME, 'r') as f:
    prunenet = pickle.load(f)

for di_offset in range(0, len(dataset), batch_size):
    print di_offset, len(dataset)
    for di in range(di_offset, min(di_offset + batch_size, len(dataset))):
        (img, label) = dataset[di]
        im_input[di - di_offset, :, :, :] = img
        labels[di] = label

    #  compute true target.
    net.blobs['data'].data[:] = im_input
    net.forward(end=layer)

    prev_layer_index =  list(net._layer_names).index(layer) - 1
    prev_layer = list(net._layer_names)[prev_layer_index]
    print '[prev_layer]', prev_layer
    im_bottom = net.blobs[prev_layer].data

    target = net.blobs[layer].data[:]
    print 'true sparsity', sparsity(sign(target))

    # compute mask.
    mask = sign(prunenet['conv'](im_bottom))
    print 'mask sparsity', sparsity(mask)

    stats_fn = np.sum(mask * (1 - sign(target))) / float(np.prod(mask.shape))
    print 'false negative', stats_fn
    stats_fp = np.sum((1 - mask) * (sign(target))) / float(np.prod(mask.shape))
    print 'false positive', stats_fp

    # compute masked layer.
    print target * mask
    net.blobs[layer].data[:] =  target * mask

    # compute final output.
    next_layer_index =  list(net._layer_names).index(layer) + 1
    next_layer = list(net._layer_names)[next_layer_index]
    print '[next_layer]', next_layer
    result = np.array(net.forward(start=next_layer)['prob'])
    label = np.argmax(result, axis=1)

    for di in range(di_offset, min(di_offset + batch_size, len(dataset))):
        gt_label = labels[di]
        if int(gt_label) == int(label[di - di_offset]):
            acc += 1
        total += 1

accuracy = float(acc) / total
print 'accuracy = ', accuracy




