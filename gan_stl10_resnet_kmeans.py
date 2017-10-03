from PIL import Image

import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')

DATA_DIR = ''  # Directory containing stl-10 dataset
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory!')

N_GPUS = 1
if N_GPUS != 1:
    raise Exception('Just 1 GPU for now!')

BATCH_SIZE = 64 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 200000 # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 6912 # Number of pixels in STL10 (resized to 48*48*3)
LR = 2e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 1000 # How frequently to calculate Inception score
KMEANS_ITERS = [2*7810, 4*7810, 6*7810, 8*7810, 10*7810] # When to recalculate labels. (BATCH_SIZE == 64 => 1 epoch ~ 2*781 iterations)
KMEANS_THRESHOLD = 2000 # Do not divide clusters with this or less samples
MAX_CLASSES = 512

CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print "WARNING! Conditional model without normalization in D might be effectively unconditional!"

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

class splitting_stl10:
    def __init__(self, init_labels=None):
        self.size = 48
        if init_labels is not None:
            self.alt_labels = init_labels
            self.n_classes = len(set(init_labels))
            self.n_cols = len(set(init_labels))
        else:
            self.alt_labels = [0] * 100000
            self.n_classes = 1
            self.n_cols = 1
        self.threshold = KMEANS_THRESHOLD

        self.images, _ = self.load_data(DATA_DIR, 'unlabeled')
        XX = np.zeros((100000, self.size, self.size, 3), dtype='uint8')
        for img in range(100000):
            XX[img] = np.asarray(Image.fromarray(self.images[img]).resize((self.size, self.size), Image.BILINEAR))
        self.images = np.transpose(XX, (0, 3, 1, 2)).reshape((-1, self.size * self.size * 3))

        samples = self.images[:100]
        lib.save_images.save_images(samples.reshape((10*10, 3, self.size, self.size)), 'real_samples.png', mod=10)

        self.train_ordered = self.generator(shuffle=False)
        self.train_gen = self.generator()

        self.tree_array = np.zeros((MAX_CLASSES, MAX_CLASSES), dtype='float32')
        self.tree_array[[range(self.n_classes), range(self.n_classes)]] = 1.0

    def generator(self, shuffle=True):

        def get_epoch():
            labels = np.array(self.alt_labels)
            rng_state = np.random.get_state()
            np.random.shuffle(self.images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)

            for i in xrange(len(self.images) / BATCH_SIZE):
                yield (self.images[i*BATCH_SIZE:(i+1)*BATCH_SIZE], labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

        def get_epoch_unshuffled():
            labels = np.array(self.alt_labels)
            for i in xrange(len(self.images) / BATCH_SIZE):
                yield (self.images[i*BATCH_SIZE:(i+1)*BATCH_SIZE], labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

        if shuffle:
            return get_epoch
        else:
            return get_epoch_unshuffled

    def load_data(self, data_dir, which_set):

        if which_set == 'train' or which_set == 'test':
            path_to_data  = os.path.join(data_dir, 'stl10_binary/' + which_set + '_X.bin')
            path_to_labels= os.path.join(data_dir, 'stl10_binary/' + which_set + '_y.bin')
        elif which_set == 'unlabeled':
            path_to_data =  os.path.join(data_dir, 'stl10_binary/' + which_set + '_X.bin')
        else:
            raise Exception(which_set + " doesn't exist")

        y = None
        if which_set is not 'unlabeled':
            with open(path_to_labels, 'rb') as f:
                y = np.fromfile(f, dtype=np.uint8) - np.uint8(1)

        with open(path_to_data, 'rb') as f:
            X = np.fromfile(f, dtype=np.uint8)

        X = np.reshape(X, (-1, 3, 96, 96))
        X = np.transpose(X, (0, 3, 2, 1))

        return X, y

    def recalc_labels(self, session, formula, iteration):
        all_features = np.zeros((100000, DIM_D))
        normformula = tf.nn.l2_normalize(formula, 0)
        for i, (images, _labs) in enumerate(self.train_ordered()):
            features = session.run(normformula, feed_dict={all_real_data_int: images, all_real_labels:_labs, _iteration:iteration})
            all_features[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE), :] = features

        np.savetxt("labels_prev_%d.txt"%iteration, self.alt_labels)

        labs = np.array(self.alt_labels)
        new_labs = np.copy(labs)
        next_class = self.n_classes
        next_col = self.n_cols
        new_classes = list()

        for i in range(self.n_classes):
            class_size = len(labs[labs == i])
            if class_size > self.threshold:
                estimator = KMeans(n_clusters=2)
                X = all_features[labs == i]
                estimator.fit(X)
                cluster_labs = np.array(estimator.labels_)
                cluster_labs[cluster_labs == 1] = next_class
                cluster_labs[cluster_labs == 0] = i
                self.tree_array[next_class] = self.tree_array[i]
                self.tree_array[next_class, next_col] = 1.0
                self.tree_array[i, next_col+1] = 1.0
                next_col += 2
                new_classes.append(next_class)
                next_class += 1
                new_labs[labs == i] = cluster_labs
            else:
                new_classes.append(None)

        self.alt_labels = list(new_labs)
        self.n_classes = next_class
        self.n_cols = next_col
        np.savetxt("labels_pos_%d.txt"%iteration, self.alt_labels)
        self.train_gen = self.generator()
        temp_gen = self.generator(shuffle=False)
        samples = [np.array([], dtype='int32').reshape(0,self.size*self.size*3)] * self.n_classes
        gen = temp_gen()
        while (min([x.shape[0] for x in samples])<10):
            images, labels = gen.next()
            arrlab = np.array(labels)
            for c in range(self.n_classes):
                samples[c] = np.concatenate([samples[c], images[np.where(arrlab==c)]])
        samples = [x[:10] for x in samples]
        all_samples = np.concatenate(samples, axis=0)
        lib.save_images.save_images(all_samples.reshape((self.n_classes*10, 3, self.size, self.size)), 'cluster_samples_%d.png'%iteration, mod=10)

        return new_classes

data_provider = splitting_stl10()

def extendVariable(var, new_classes):
    updated = var
    for i, new_class in enumerate(new_classes):
        if new_class is not None:
            updated = tf.scatter_update(updated, [new_class], tf.expand_dims(var[i], 0))
    return updated

def nonlinearity(x):
    return tf.nn.relu(x)

tree_var = tf.Variable(data_provider.tree_array)

def Normalize(name, inputs,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm, 
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        raise RuntimeError("TODO: check")
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            result, _, _ = lib.ops.cond_batchnorm.BatchnormB(name,[0,2,3],inputs,labels=tf.nn.embedding_lookup(tree_var, labels),n_labels=MAX_CLASSES, n_start_labels=data_provider.n_classes)
            return result
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output

def OptimizedResBlockDisc1(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, 6*6*DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 6, 6])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 48, 48])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, MAX_CLASSES, output, extensible=True)
        return output_wgan, output_acgan, output
    else:
        return output_wgan, None, output

with tf.Session() as session:

    _iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    fake_data_splits = []
    for i, device in enumerate(DEVICES):
        #with tf.device(device):
            fake_data_splits.append(Generator(BATCH_SIZE/len(DEVICES), labels_splits[i]))

    all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    all_real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

    DEVICES_B = DEVICES[:len(DEVICES)/2]
    DEVICES_A = DEVICES[len(DEVICES)/2:]

    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []
    for i, device in enumerate(DEVICES_A):
        #with tf.device(device):
            real_and_fake_data = tf.concat([
                all_real_data_splits[i],
                all_real_data_splits[len(DEVICES_A)+i],
                fake_data_splits[i],
                fake_data_splits[len(DEVICES_A)+i]
            ], axis=0)
            real_and_fake_labels = tf.concat([
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i],
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i]
            ], axis=0)
            disc_all, disc_all_acgan, disc_all_output = Discriminator(real_and_fake_data, real_and_fake_labels)
            disc_real = disc_all[:BATCH_SIZE/len(DEVICES_A)]
            disc_fake = disc_all[BATCH_SIZE/len(DEVICES_A):]
            disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
            if CONDITIONAL and ACGAN:
                disc_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], labels=real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)])
                ))
                disc_acgan_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], dimension=1)),
                            real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)]
                        ),
                        tf.float32
                    )
                ))
                disc_acgan_fake_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[BATCH_SIZE/len(DEVICES_A):], dimension=1)),
                            real_and_fake_labels[BATCH_SIZE/len(DEVICES_A):]
                        ),
                        tf.float32
                    )
                ))


    for i, device in enumerate(DEVICES_B):
        #with tf.device(device):
            real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A)+i]], axis=0)
            fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
            labels = tf.concat([
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i],
            ], axis=0)
            alpha = tf.random_uniform(
                shape=[BATCH_SIZE/len(DEVICES_A),1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates, labels)[0], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
            disc_costs.append(gradient_penalty)
            ## Dataset to discriminator coding
            disc_codification = Discriminator(real_data, labels)[2]

    disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)
    if CONDITIONAL and ACGAN:
        disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
        disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
        disc_cost = disc_wgan + (ACGAN_SCALE*disc_acgan)
    else:
        disc_acgan = tf.constant(0.)
        disc_acgan_acc = tf.constant(0.)
        disc_acgan_fake_acc = tf.constant(0.)
        disc_cost = disc_wgan

    disc_params = lib.params_with_name('Discriminator.')

    if DECAY:
        decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
    else:
        decay = 1.

    n_classes_var = tf.Variable(data_provider.n_classes, dtype=tf.float32)
    gen_costs = []
    gen_acgan_costs = []
    for device in DEVICES:
        #with tf.device(device):
            n_samples = GEN_BS_MULTIPLE * BATCH_SIZE / len(DEVICES)
            fake_labels = tf.cast(tf.random_uniform([n_samples])*n_classes_var, tf.int32)
            if CONDITIONAL and ACGAN:
                disc_fake, disc_fake_acgan, disc_fake_output = Discriminator(Generator(n_samples,fake_labels), fake_labels)
                gen_costs.append(-tf.reduce_mean(disc_fake))
                gen_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                ))
            else:
                gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels), fake_labels)[0]))
    gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
    if CONDITIONAL and ACGAN:
        gen_cost += (ACGAN_SCALE_G*(tf.add_n(gen_acgan_costs) / len(DEVICES)))


    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # Function for generating samples (random)
    def generate_image_b(frame, n_per_class):
        fake_labels_100 = tf.cast(tf.random_uniform([100]) * data_provider.n_classes, tf.int32)
        samples_100 = Generator(100, fake_labels_100)
        samples = [np.array([], dtype='int32').reshape(0,OUTPUT_DIM)] * data_provider.n_classes

        while (min([x.shape[0] for x in samples])<n_per_class):
            images, labels = session.run((samples_100, fake_labels_100))
            arrlab = np.array(labels)
            for c in range(data_provider.n_classes):
                samples[c] = np.concatenate([samples[c], images[np.where(arrlab==c)]])
        samples = [x[:n_per_class] for x in samples]
        all_samples = np.concatenate(samples, axis=0)
        all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
        lib.save_images.save_images(all_samples.reshape((data_provider.n_classes*n_per_class, 3, 48, 48)), 'random_samples_{}.png'.format(frame))

    # Function for calculating inception score
    def get_inception_score(n, iter):
        fake_labels_100 = tf.cast(tf.random_uniform([100]) * data_provider.n_classes, tf.int32)
        samples_100 = Generator(100, fake_labels_100)
        all_samples = []
        for i in xrange(n/100):
            samples = session.run(samples_100)
            all_samples.append(samples)
        first_samples = all_samples[0]
        first_samples = ((first_samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(first_samples.reshape((100, 3, 48, 48)),
                                    'inception_score_samples%d.png'%iter)
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 48, 48)).transpose(0,2,3,1)
        return lib.inception_score.get_inception_score(list(all_samples))

    def inf_train_gen():
        while True:
            for images,_labels in data_provider.train_gen():
                yield images,_labels


    for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print "{} Params:".format(name)
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g == None:
                print "\t{} ({}) [no grad!]".format(v.name, shape_str)
            else:
                print "\t{} ({})".format(v.name, shape_str)
        print "Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        )

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()

    min_iscore = 8.0
    saver = tf.train.Saver(max_to_keep=None)
    for iteration in xrange(ITERS):
        start_time = time.time()
        sys.stdout.flush()

        if iteration > 0:
            _ = session.run([gen_train_op], feed_dict={_iteration:iteration})

        for i in xrange(N_CRITIC):
            _data,_labels = gen.next()
            if CONDITIONAL and ACGAN:
                _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = session.run([disc_cost, disc_wgan, disc_acgan, disc_acgan_acc, disc_acgan_fake_acc, disc_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})
            else:
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})

        lib.plot.plot('cost', _disc_cost)
        if CONDITIONAL and ACGAN:
            lib.plot.plot('wgan', _disc_wgan)
            lib.plot.plot('acgan', _disc_acgan)
            lib.plot.plot('acc_real', _disc_acgan_acc)
            lib.plot.plot('acc_fake', _disc_acgan_fake_acc)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY-1:
            inception_score = get_inception_score(50000, iteration)
            lib.plot.plot('inception_50k', inception_score[0])
            lib.plot.plot('inception_50k_std', inception_score[1])
            if iteration>10000 and inception_score[0]>min_iscore:
                min_iscore = inception_score[0]
                saver.save(session, "logs/model_best", global_step=iteration)
                inception_score_2 = get_inception_score(50000, iteration)
                print("BEST SCORE RECALC:")
                print(inception_score_2)

        # Generate samples every 100 iters
        if iteration % 100 == 99:
            generate_image_b(iteration, 10)

        if iteration in KMEANS_ITERS:
            prev_clusters = data_provider.n_classes
            print("KMEANS...")
            saver.save(session, "logs/model_prev", global_step=iteration)
            new_classes = data_provider.recalc_labels(session, disc_codification, iteration)
            for (varname, var) in lib.get_extensible_params():
                session.run(extendVariable(var, new_classes))
                if "Generator" in varname:
                    slots = gen_opt.get_slot_names()
                    for slot in slots:
                        adamvar = gen_opt.get_slot(var, slot)
                        session.run(extendVariable(adamvar, new_classes))
                if "Discriminator." in varname:
                    slots = disc_opt.get_slot_names()
                    for slot in slots:
                        adamvar = disc_opt.get_slot(var, slot)
                        session.run(extendVariable(adamvar, new_classes))

            gen = inf_train_gen()
            print("KMEANS done. N_classes = %d"%data_provider.n_classes)

            pos_clusters = data_provider.n_classes
            if prev_clusters!=pos_clusters:
                print("Continue training -- prev: %d pos: %d"%(prev_clusters, pos_clusters))
                session.run(n_classes_var.assign(pos_clusters))
                session.run(tree_var.assign(data_provider.tree_array))
                np.savetxt("tree_%d.txt"%iteration, data_provider.tree_array, fmt='%d')
            else:
                print("Continue training -- prev: %d pos: %d" % (prev_clusters, pos_clusters))

        if (iteration < 50) or (iteration % 1000 == 999):
            lib.plot.flush()

        lib.plot.tick()
