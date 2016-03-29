# merge conv1 into the network and get *training*, *test* accuracy.
import os
os.environ['SKIP_TRAINING_DATA'] = ' ' # skip training data.
from common import *
import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array

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

    net.blobs['data'].data[:] = im_input
    result = net.forward()['prob']
    label = np.argmax(result, axis=1)

    for di in range(di_offset, min(di_offset + batch_size, len(dataset))):
        gt_label = labels[di]
        if int(gt_label) == int(label[di - di_offset]):
            acc += 1
        total += 1

accuracy = float(acc) / total
print 'accuracy = ', accuracy




