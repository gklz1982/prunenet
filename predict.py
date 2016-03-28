# experiment script.
# do prediction.
import matplotlib
matplotlib.use('Agg')
import caffe
from caffe.io import oversample, resize_image
from PIL import Image
import numpy as np

MODEL_FILE = 'models/bvlc_reference_caffenet/deploy.prototxt'
WEIGHTS_FILE = 'models/bvlc_reference_caffenet/caffenet.caffemodel'
MEAN_FILE = 'data/imagenet/mean.lmdb'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( MEAN_FILE , 'rb' ).read()
blob.ParseFromString(data)
mean_arr = np.array( caffe.io.blobproto_to_array(blob) )[0]

img = Image.open('data/imagenet/train/n03085013/n03085013_2061.JPEG')
img = np.array(img, dtype='float') # convert to single.
img = resize_image(img, (256, 256))
img = np.array([img[:, :, 0], img[:, :, 1], img[:, :, 2]])
img = img - mean_arr # substract mean.
# central crop.
(h, w) = img.shape[1:]
(th, tw) = (227, 227)
(ch, cw) = (int(h / 2), int(w / 2))
(cth, ctw) = (int(th / 2), int(tw / 2))
img = img[:, ch-cth:ch-cth+th, cw-ctw:cw-ctw+tw]
im_input = img[np.newaxis, :, :, :]

net = caffe.Net(MODEL_FILE, WEIGHTS_FILE)

net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[:] = im_input

result = net.forward()['prob']
prob = result[0, :, 0, 0]

def sparsity(blob_data):
    return float(np.sum(np.abs(blob_data) > 0)) / np.prod(blob_data.shape)

for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']:
    print '[sparsity]', layer, sparsity(net.blobs[layer].data)




















