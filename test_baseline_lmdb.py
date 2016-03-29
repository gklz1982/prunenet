# merge conv1 into the network and get *training*, *test* accuracy.
import os
os.environ['SKIP_TRAINING_DATA'] = ' ' # skip training data.
from common import *
import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array

batch_size = 32
im_input = np.zeros((batch_size, 3, th, tw))

# load original model.
net = caffe.Net(MODEL_FILE, WEIGHTS_FILE)
net.set_mode_gpu()
net.blobs['data'].reshape(*im_input.shape)

acc = 0
total = 0

data_list = test_list
data_label = test_label

for di_offset in range(0, len(data_list), batch_size):
    print di_offset, len(data_list)
    for di in range(di_offset, min(di_offset + batch_size, len(data_list))):
        img = data_list[di]
        im_input[di - di_offset, :, :, :] = img

    net.blobs['data'].data[:] = im_input
    result = net.forward()['prob']
    label = np.argmax(result, axis=1)

    for di in range(di_offset, min(di_offset + batch_size, len(data_list))):
        gt_label = data_label[di]
        if int(gt_label) == int(label[di - di_offset]):
            acc += 1
        total += 1

accuracy = float(acc) / total
print 'accuracy = ', accuracy




