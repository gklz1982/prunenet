# experiment script.
# prune conv3.
from common import *

batch_size = 32
RESULT_NAME = 'stride'

layer_name = 'conv3'
bottom_blob = 'norm2'
top_blob = 'conv3'


npr.seed(0)

im_input = sample_minibatch(train_set, batch_size).astype(np.float32)

net = caffe.Net(MODEL_FILE, WEIGHTS_FILE)
net.set_mode_gpu()
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[:] = im_input



with Timer('full forward'):
    result = net.forward()['prob']
    print 'result shape', result.shape
    im_bottom = net.blobs[bottom_blob].data
    im_target = net.blobs[top_blob].data
    print 'sparsity', sparsity(im_target)
    im_target = sign(im_target).astype(np.float32)
    # im_target = im_target.astype(np.float32)

# training.

with Timer('create tensor network'):
    filter_size = (1, 1)
    input_dim = 256
    output_dim = 384
    stride = 1
    pad = 1

    drop_factor = 3

    eps = np.float32(1.) # hinge-loss parameter.
    lam = np.float32(1.) # sparsity penalty.
    lr = np.float32(1e-3) # learning rate.

    print 'bottom shape', im_bottom.shape
    print 'target shape', im_target.shape

    std_dev = np.sqrt(2. / (input_dim * filter_size[0] * filter_size[1]))
    W_init = std_dev * np.random.randn(output_dim, input_dim, filter_size[0], filter_size[1]).astype(np.float32)
    W = theano.shared(value=W_init, name='W')

    # initialize bias vector b of size (output_dim, 1)
    b_init = np.zeros((output_dim)).astype(np.float32)
    b = theano.shared(value=b_init, name='b')

    inputs = T.tensor4('input')
    inputs_padded = T.zeros((im_bottom.shape[0], im_bottom.shape[1], im_bottom.shape[2] + pad * 2, im_bottom.shape[3] + pad * 2),
                     dtype=np.float32)
    inputs_padded = T.set_subtensor(inputs_padded[:, :, pad:-pad, pad:-pad], inputs)

    inputs_sampled = max_pool_2d(inputs_padded, ds=(drop_factor, drop_factor), ignore_border=False)

    conv_out = T.nnet.conv.conv2d(inputs_sampled, W, #border_mode='full',
                                  subsample=(stride, stride))

    conv_out = T.nnet.abstract_conv.bilinear_upsampling(conv_out, ratio=drop_factor,
                                                        batch_size=batch_size,
                                                        num_input_channels=output_dim,
                                                        use_1D_kernel=False)
    conv_out += b.dimshuffle('x', 0, 'x', 'x')
    conv_out = conv_out[:, :, :im_target.shape[2], :im_target.shape[3]]

    with Timer('compiling fprop function'):
        conv = theano.function([inputs], conv_out)

    result = conv(im_bottom)

    print 'result_shape', result.shape

    targets = T.tensor4('target')

    conv_out_sigmoid = T.nnet.sigmoid(conv_out)

    #loss_fp = -T.sum(targets * T.log(conv_out_sigmoid))
    #loss_sparsity = -T.sum((1 - targets) * T.log(1-conv_out_sigmoid))
    loss_fp = T.sum(targets * T.maximum(np.float32(0.), eps - conv_out)) / T.sum(targets) * np.float32(np.prod(im_target.shape) / 2)
    loss_sparsity = T.sum((1 - targets) * T.maximum(np.float32(0.), eps + conv_out)) / T.sum(np.float32(1.) - targets) * np.float32(np.prod(im_target.shape) / 2)


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
            net.blobs['data'].data[:] = im_input
            im_bottom = net.blobs[bottom_blob].data
            result = net.forward()['prob']

        with Timer('backpropagation'):
            im_target = net.blobs[top_blob].data
            im_target = sign(im_target)

            loss = bprop(im_bottom, im_target)

            print 'loss', loss / np.prod(result.shape)

        with Timer('test on validation set'):
            if it % 100 == 0:
                target = conv(im_bottom)
                print 'target', target[0, 0, :, :]
                print 'im_target', im_target[0, 0, :, :]


                stats_sparsity = sparsity(np.maximum(0, target))
                print 'sparsity', stats_sparsity
                stats_fp = np.sum(sign(im_target) * (1 - sign(target))) / float(np.prod(im_target.shape))
                print 'false positive', stats_fp
                print
                stats_fn = np.sum(sign(1 - im_target) * (sign(target))) / float(np.prod(im_target.shape))
                print 'false negative', stats_fn

                sparsity_history.append(stats_sparsity)
                fp_history.append(stats_fn)


with open('%(name)s-%(layer_name)s.model'
            % dict(name = RESULT_NAME,
                   layer_name = layer_name),
          'w') as f:
    with Timer('dumping model to %s' % RESULT_NAME):
        pickle.dump({
            'conv': conv,
            #'bprop': bprop,
            'W': W.get_value(),
            'b': b.get_value(),
            'trueW': net.params[layer_name][0].data,
            'sparsity': sparsity_history,
            'fn': fp_history
        }, f)

















