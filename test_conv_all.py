import os
os.environ['SKIP_TRAINING_DATA'] = ' ' # skip training data.
from common import *
import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array


name = os.environ.get('name')
if not name:
    name = 'stride'

print '[environ] name = ', name

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
droplayer = {}

for layer in layers:
    with open('%s-%s.model' % (name, layer), 'r') as f:
        droplayer[layer] = pickle.load(f)


batch_size = 32

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

for di_offset in range(0, len(dataset), batch_size):
    print di_offset, len(dataset)
    for di in range(di_offset, min(di_offset + batch_size, len(dataset))):
        (img, label) = dataset[di]
        im_input[di - di_offset, :, :, :] = img
        labels[di] = label

    #  compute true target.
    blob_names = net.blobs.keys()

    net.blobs['data'].data[:] = im_input

    start_layer = None

    for layer in layers:
        net.forward(start=start_layer, end=layer)

        prev_blob_index =  blob_names.index(layer) - 1
        prev_blob = blob_names[prev_blob_index]
        im_bottom = net.blobs[prev_blob].data

        target = net.blobs[layer].data[:]

        # compute mask.
        mask = sign(droplayer[layer]['conv'](im_bottom))

        # compute masked layer.
        net.blobs[layer].data[:] =  target * mask

        # compute final output.
        next_layer_index =  list(net._layer_names).index(layer) + 1
        start_layer = list(net._layer_names)[next_layer_index]

    result = np.array(net.forward(start=start_layer)['prob'])
    label = np.argmax(result, axis=1)

    for di in range(di_offset, min(di_offset + batch_size, len(dataset))):
        gt_label = labels[di]
        if int(gt_label) == int(label[di - di_offset]):
            acc += 1
        total += 1

accuracy = float(acc) / total
print 'accuracy = ', accuracy




