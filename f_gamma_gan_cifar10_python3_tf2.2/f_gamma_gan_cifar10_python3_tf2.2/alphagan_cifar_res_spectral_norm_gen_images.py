#""WGAN-GP ResNet for CIFAR-10""

import os, sys
username='mcgregor'
sys.path.append('/home/'+username+'/CUMGAN/cumulant_gan_cifar10_python3_tf2.2/')

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot
import csv


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sklearn.datasets

import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = '/home/'+username+'/CUMGAN/cumulant_gan_cifar10_python3_tf2.2/data/cifar-10-batches-py'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

N_GPUS = 1
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

ITERS=int(sys.argv[1]) # How many iterations to train for
alpha = float(sys.argv[2])  # cumgan parameters
rev = int(sys.argv[3])  # cumgan parameters
itr=int(sys.argv[4])    # total number of runs
LR=float(sys.argv[5])   # Initial learning rate: original value=2e-4
sess_name = sys.argv[6] # name of the session run


BATCH_SIZE = 64 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
DIM_G = 256 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic? This doesn't do anything at the moment.
OUTPUT_DIM = 3072 # Number of pixels in cifar10 (32*32*3)
DECAY = True # Whether to decay LR over learning
INCEPTION_FREQUENCY = 100 # How frequently to calculate Inception score
j=0

CONDITIONAL = False # Whether to train a conditional or unconditional model
ACGAN = False # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print ("WARNING! Conditional model without normalization in D might be effectively unconditional!")

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm, 
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
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
    
def MeanPoolConv_spec_norm(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D_spec_norm(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output
    
def ConvMeanPool_spec_norm(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D_spec_norm(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output
    
def UpsampleConv_spec_norm(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D_spec_norm(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
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
    
def ResidualBlock_spec_norm(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D_spec_norm, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool_spec_norm, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool_spec_norm
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv_spec_norm, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv_spec_norm
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D_spec_norm, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D_spec_norm
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D_spec_norm, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D_spec_norm, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)    
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

def OptimizedResBlockDisc1_spec_norm(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D_spec_norm, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool_spec_norm, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv_spec_norm
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)    
    output = nonlinearity(output)            
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output
    



def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

#use spectral normalization in discriminator
def Discriminator(inputs, labels):
    #Cifar 10: 32x32 images, 3 channels
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1_spec_norm(output)
    output = ResidualBlock_spec_norm('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down')
    output = ResidualBlock_spec_norm('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None)
    output = ResidualBlock_spec_norm('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None)


    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output = lib.ops.linear.Linear_spec_norm('Discriminator.Output', DIM_D, 1, output)
    output = tf.reshape(output, [-1])
    return output, None

#for alpha>1
def f_alpha_star(y,alpha):
    return tf.math.pow(tf.nn.relu(y),alpha/(alpha-1.0))*tf.math.pow((alpha-1.0),alpha/(alpha-1.0))/alpha+1/(alpha*(alpha-1.0))


if not os.path.exists('cifar_resnet_sn/'+str(sess_name)+'/'):
    os.makedirs('cifar_resnet_sn/'+str(sess_name)+'/')

with tf.Session() as session:

    _iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    inception_file = 'cifar_resnet_sn/'+str(sess_name)+'/inception_scores_alpha_' + str(alpha) + '_rev_' + str(rev)+ '_set'+str(j+itr) + '.csv'

    if os.path.isfile(inception_file):
        inception_scores = []
        with open(inception_file, "r") as output:
            reader = csv.reader(output, lineterminator='\n')
            for val in reader:
                inception_scores.append(val[0])
    else:
        inception_scores = []

    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    fake_data_splits = []
    for i, device in enumerate(DEVICES):
        with tf.device(device):
            fake_data_splits.append(Generator(int(BATCH_SIZE/len(DEVICES)), labels_splits[i]))

    all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    all_real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)
    DEVICES_B = DEVICES[:int(len(DEVICES)/2)]
    DEVICES_A = DEVICES[int(len(DEVICES)/2):]
    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []



    for i, device in enumerate(DEVICES_A):
        with tf.device(device):
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
            disc_all, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels)
            disc_real = disc_all[:int(BATCH_SIZE/len(DEVICES_A))]
            disc_fake = disc_all[int(BATCH_SIZE/len(DEVICES_A)):]

           #losses
            if alpha==0:    # gradient penalty Standard WGAN loss
                disc_cost_fake = tf.reduce_mean(disc_fake)
                disc_cost_real = tf.reduce_mean(disc_real)
                disc_cost_tot=disc_cost_real-disc_cost_fake
            elif alpha==1:
                if rev == 1:
                    disc_cost_fake = tf.reduce_mean(disc_fake)
                    disc_cost_real = tf.math.log(tf.reduce_mean(tf.math.exp(disc_real)))
                    disc_cost_tot=disc_cost_fake-disc_cost_real
                elif rev == 0:
                    disc_cost_real = tf.reduce_mean(disc_real)
                    disc_cost_fake = tf.math.log(tf.reduce_mean(tf.math.exp(disc_fake)))
                    disc_cost_tot=disc_cost_real-disc_cost_fake
            
            elif alpha==-1: #this is the code for alpha=infinity
                if rev == 1:
                    disc_cost_fake = tf.reduce_mean(disc_fake)
                    disc_cost_real = tf.reduce_mean(tf.nn.relu(disc_real))
                    disc_cost_tot=disc_cost_fake-disc_cost_real
                elif rev == 0:
                    disc_cost_real = tf.reduce_mean(disc_real)
                    disc_cost_fake = tf.reduce_mean(tf.nn.relu(disc_fake))
                    disc_cost_tot=disc_cost_real-disc_cost_fake
            
            else: #reverse generalized alpha GAN
                if rev == 1:
                    disc_cost_fake = tf.reduce_mean(disc_fake)
                    disc_cost_real = tf.reduce_mean(f_alpha_star(disc_real,alpha))
                    disc_cost_tot=disc_cost_fake-disc_cost_real
                elif rev == 0:
                    disc_cost_real = tf.reduce_mean(disc_real)
                    disc_cost_fake = tf.reduce_mean(f_alpha_star(disc_fake,alpha))
                    disc_cost_tot=disc_cost_real-disc_cost_fake

            disc_costs.append(-disc_cost_tot)


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
        with tf.device(device):
            real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A)+i]], axis=0)
            fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
            labels = tf.concat([
                labels_splits[i], 
                labels_splits[len(DEVICES_A)+i],
            ], axis=0)
            

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

    gen_costs = []
    gen_acgan_costs = []
    for device in DEVICES:
        with tf.device(device):
            n_samples = GEN_BS_MULTIPLE * int(BATCH_SIZE / len(DEVICES))
            fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
            if CONDITIONAL and ACGAN:
                disc_fake, disc_fake_acgan = Discriminator(Generator(n_samples,fake_labels), fake_labels)
                gen_costs.append(-tf.reduce_mean(disc_fake))
                gen_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                ))
            else:
                disc_fake = Discriminator(Generator(n_samples, fake_labels), fake_labels)[0]
                
                #losses
                if alpha==0:    # gradient penalty Standard WGAN loss
                    gen_c=-tf.reduce_mean(disc_fake)
                elif alpha==1:
                    if rev == 1:
                        gen_c=tf.reduce_mean(disc_fake)
                    elif rev == 0:
                        gen_c=-tf.math.log(tf.reduce_mean(tf.math.exp(disc_fake)))
                
                elif alpha==-1: #this is the code for alpha=infinity
                    if rev == 1:
                        gen_c=tf.reduce_mean(disc_fake)
                    elif rev == 0:
                        gen_c=- tf.reduce_mean(tf.nn.relu(disc_fake))
                
                else: #reverse generalized alpha GAN
                    if rev == 1:
                        gen_c=tf.reduce_mean(disc_fake)
                    elif rev == 0:
                        gen_c=- tf.reduce_mean(f_alpha_star(disc_fake,alpha))

                 
                

                gen_costs.append(gen_c)

                #gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels), fake_labels)[0]))

    gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
    if CONDITIONAL and ACGAN:
        gen_cost += (ACGAN_SCALE_G*(tf.add_n(gen_acgan_costs) / len(DEVICES)))


    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # Saving and loading checkpoints

    def save_checkpoint(checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)  
        saver.save(session, os.path.join(checkpoint_dir, 'cumgan'), global_step=step)

    def load_checkpoint(checkpoint_dir):
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            return step
        else:
            return 0

    saver = tf.train.Saver(max_to_keep=1)

    # Function for generating samples
    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        #lib.save_images.save_images(samples.reshape((100, 3, 32, 32)), 'samples/cifar10_resnet/samples_{}.png'.format(frame))

    # Function for calculating inception score
    fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
    samples_100 = Generator(100, fake_labels_100)
    def get_inception_score(n):
        all_samples = []
        for i in range(int(n/100)):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
        return lib.inception_score.get_inception_score(list(all_samples))

    train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR)
    def inf_train_gen():
        while True:
            for images,_labels in train_gen():
                yield images,_labels


    for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print ("{} Params:".format(name))
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g == None:
                print ("\t{} ({}) [no grad!]".format(v.name, shape_str))
            else:
                print ("\t{} ({})".format(v.name, shape_str))
        print ("Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        ))

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()

    iteration = load_checkpoint('cifar_resnet_sn/'+str(sess_name)+'/checkpoints/')

    all_samples = []
    for i in range(500):
        all_samples.append(session.run(samples_100))

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    gen_image_file = 'cifar_resnet_sn/'+str(sess_name)+'/gen_images_50k_alpha_' + str(alpha) + '_rev_' + str(rev)+ '_set'+str(j+itr) +'_iteration'+ str(iteration)+ '.npy'
    np.save(gen_image_file, all_samples)
    print(""" ========================================== """)
    print(""" Finish saving .npy files""")
    print(""" ========================================== """)
