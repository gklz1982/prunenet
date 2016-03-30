# experiment script.
# prune conv1.
import matplotlib
import dill as pickle
matplotlib.use('Agg')
import caffe
from caffe.io import oversample, resize_image
from scipy.misc import imresize
from PIL import Image
from utils import *
from theano.tensor.nnet import conv2d
import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
from optimizers import Adam
from common import *
from theano.tensor.nnet.abstract_conv import bilinear_kernel_2D

batch_size = 32
RESULT_NAME = 'conv1-stride.model'

npr.seed(0)

im_input = sample_minibatch(train_set, batch_size).astype(np.float32)

net = caffe.Net(MODEL_FILE, WEIGHTS_FILE)
net.set_mode_gpu()
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[:] = im_input



with Timer('full forward'):
    result = net.forward()['prob']
    print 'result shape', result.shape
    im_target = net.blobs['conv1'].data
    print 'sparsity', sparsity(im_target)
    im_target = sign(im_target).astype(np.float32)
    # im_target = im_target.astype(np.float32)

# training.

with Timer('create tensor network'):
    filter_size = (6, 6) # down-sampled (11, 11)
    input_dim = 3
    output_dim = 96
    stride = 4 # down-sampled (4, 4)

    eps = np.float32(1.) # hinge-loss parameter.
    lam = np.float32(1.) # sparsity penalty.
    lr = np.float32(1e-3) # learning rate.

    std_dev = np.sqrt(2. / (input_dim * filter_size[0] * filter_size[1]))
    W_init = std_dev * np.random.randn(output_dim, input_dim, filter_size[0], filter_size[1]).astype(np.float32)
    W = theano.shared(value=W_init, name='W')

# initialize bias vector b of size (output_dim, 1)
    b_init = np.zeros((output_dim)).astype(np.float32)
    b = theano.shared(value=b_init, name='b')

    inputs = T.tensor4('input')
    inputs_sampled = max_pool_2d(inputs, ds=(2, 2), ignore_border=False)

    conv_out = T.nnet.conv.conv2d(inputs_sampled, W, #border_mode='full',
                                  subsample=(stride, stride),
                                  image_shape=(im_input.shape[0], im_input.shape[1],
                                               int(np.ceil(im_input.shape[2] / 2.)),
                                               int(np.ceil(im_input.shape[3] / 2.))
                                               ),
                                  filter_shape=(output_dim, input_dim) + filter_size)
    conv_out = T.nnet.abstract_conv.bilinear_upsampling(conv_out, ratio=2,
                                                        batch_size=batch_size,
                                                        num_input_channels=output_dim,
                                                        use_1D_kernel=False)
    conv_out += b.dimshuffle('x', 0, 'x', 'x')
    print 'target shape', im_target.shape[2], im_target.shape[3]
    conv_out = conv_out[:, :, :im_target.shape[2], :im_target.shape[3]]

    with Timer('compiling fprop function'):
        conv = theano.function([inputs], conv_out)

    result = conv(im_input)

    print 'result_shape', result.shape

    targets = T.tensor4('target')

    conv_out_sigmoid = T.nnet.sigmoid(conv_out)

    #loss_fp = -T.sum(targets * T.log(conv_out_sigmoid))
    #loss_sparsity = -T.sum((1 - targets) * T.log(1-conv_out_sigmoid))
    loss_fp = T.sum(targets * T.maximum(np.float32(0.), eps - conv_out))
    loss_sparsity = T.sum((1 - targets) * T.maximum(np.float32(0.), eps + conv_out))


    loss = loss_fp + lam * loss_sparsity

    # loss = T.sum((T.maximum(np.float32(0.), conv_out) - targets) ** 2) # square loss.

    updates = Adam(loss, [W, b], alpha=lr)

    with Timer('compiling bprop function'):
        bprop = theano.function(inputs=[inputs, targets], outputs=loss, updates=updates)

    num_iter = 4000


    sparsity_history = []
    fp_history = []


    for it in range(num_iter):
        print '------------------'
        print 'iter', it
        with Timer('sample minibatch'):
            im_input = sample_minibatch(train_set, batch_size).astype(np.float32)

        with Timer('backpropagation'):
            net.blobs['data'].data[:] = im_input
            result = net.forward()['prob']
            im_target = net.blobs['conv1'].data
            im_target = sign(im_target)

            loss = bprop(im_input, im_target[:, :, :, :])

            print 'loss', loss / np.prod(result.shape)

        with Timer('test on validation set'):
            if it % 100 == 0:
                target = conv(im_input)
                print 'target', target[0, 0, :, :]
                print 'im_target', im_target[0, 0, :, :]


                stats_sparsity = sparsity(np.maximum(0, target))
                print 'sparsity', stats_sparsity
                stats_fn = np.sum(sign(im_target) * (1 - sign(target))) / float(np.prod(im_target.shape))
                print 'false negative', stats_fn
                print
                stats_fp = np.sum(sign(1 - im_target) * (sign(target))) / float(np.prod(im_target.shape))
                print 'false negative', stats_fn

                sparsity_history.append(stats_sparsity)
                fp_history.append(stats_fn)


with open(RESULT_NAME, 'w') as f:
    with Timer('dumping model to %s' % RESULT_NAME):
        pickle.dump({
            'conv': conv,
            'bprop': bprop,
            'W': W.get_value(),
            'b': b.get_value(),
            'trueW': net.params['conv1'][0].data,
            'sparsity': sparsity_history,
            'fn': fp_history
        }, f)

















