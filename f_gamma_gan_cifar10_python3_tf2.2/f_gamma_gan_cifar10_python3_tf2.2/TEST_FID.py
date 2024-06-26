
import numpy as np
import os
import urllib
import gzip
import pickle
import sys
import csv

username='mcgregor'
sys.path.append('/home/'+username+'/CUMGAN/cumulant_gan_cifar10_python3_tf2.2/')

gen_data_dir=sys.argv[1]
fid_file=sys.argv[2]




DATA_DIR = '/home/'+username+'/CUMGAN/cumulant_gan_cifar10_python3_tf2.2/data/cifar-10-batches-py'
HEIGHT=WIDTH=32
DATA_DIM=HEIGHT*WIDTH*3
BATCH_SIZE=50

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding='latin1')
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0).reshape([-1,3,32,32]).transpose([0,2,3,1]).reshape([-1,32*32*3])
    labels = np.concatenate(all_labels, axis=0)
        
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(len(images) // batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir), 
        cifar_generator(['test_batch'], batch_size, data_dir)
    )
def inf_gen(MODE='TRAIN', BATCH_SIZE=BATCH_SIZE):
  if MODE=='TRAIN':
    train_gen, _ = load(BATCH_SIZE, data_dir=DATA_DIR)
    while True:
        for original_images, labels in train_gen():
          yield 2./255*original_images-1,labels
  elif MODE=='TEST':
    _, test_gen = load(BATCH_SIZE, data_dir=DATA_DIR)
    while True:
        for original_images, labels in test_gen():
          yield 2./255*original_images-1,labels


train_gen=inf_gen('TRAIN')
test_gen=inf_gen('TEST')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
from tensorflow.python.ops import array_ops
# pip install tensorflow-gan
import tensorflow_gan as tfgan

session=tf.compat.v1.InteractiveSession()
# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 50

# Run images through Inception.
inception_images = tf.compat.v1.placeholder(tf.float32, [None, 3, None, None], name = 'inception_images')
activations1 = tf.compat.v1.placeholder(tf.float32, [None, None], name = 'activations1')
activations2 = tf.compat.v1.placeholder(tf.float32, [None, None], name = 'activations2')
fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_FINAL_POOL = 'pool_3'

def inception_activations(images = inception_images, num_splits = 1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.compat.v1.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    activations = tf.map_fn(
        fn = tfgan.eval.classifier_fn_from_tfhub(INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations

activations =inception_activations()

def get_inception_activations(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    act = np.zeros([inps.shape[0], 2048], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] / 255. * 2 - 1
        act[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(activations, feed_dict = {inception_images: inp})
    return act

def activations2distance(act1, act2):
    return session.run(fcd, feed_dict = {activations1: act1, activations2: act2})
        
def get_fid(images1, images2):
    session=tf.get_default_session()
    assert(type(images1) == np.ndarray)
    assert(len(images1.shape) == 4)
    assert(images1.shape[1] == 3)
    assert(np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
    assert(type(images2) == np.ndarray)
    assert(len(images2.shape) == 4)
    assert(images2.shape[1] == 3)
    assert(np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
    assert(images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
    print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(images1)
    act2 = get_inception_activations(images2)
    fid = activations2distance(act1, act2)
    print('FID calculation time: %f s' % (time.time() - start_time))
    return fid

def train_test_sets_fid(n, gen1, gen2):
    all_real_samples = np.zeros([n//BATCH_SIZE*BATCH_SIZE,DATA_DIM],dtype=np.uint8)
    all_fake_samples = np.zeros([n//BATCH_SIZE*BATCH_SIZE,DATA_DIM],dtype=np.uint8)
    for i in range(int(np.ceil(float(n)/BATCH_SIZE))):
      all_real_samples[i*BATCH_SIZE:(i+1)*BATCH_SIZE]=((next(gen1)[0]+1)/2*255).astype(np.uint8)
      all_fake_samples[i*BATCH_SIZE:(i+1)*BATCH_SIZE]=((next(gen2)[0]+1)/2*255).astype(np.uint8)
    gen_samples = np.load(gen_data_dir)
    #gen_samples = ((gen_samples+1.)*(255./2)).astype('int32')
    sample_size=min(all_fake_samples.shape[0],all_real_samples.shape[0])
    return get_fid(all_real_samples[:sample_size].reshape([-1,HEIGHT,WIDTH,3]).transpose([0,3,1,2]),gen_samples[:sample_size].transpose([0,3,1,2]))    
    #return get_fid(all_real_samples[:sample_size].reshape([-1,HEIGHT,WIDTH,3]).transpose([0,3,1,2]),all_fake_samples[:sample_size].reshape([-1,HEIGHT,WIDTH,3]).transpose([0,3,1,2]))

# FID between training set and test set
_fid = train_test_sets_fid(50000, train_gen, test_gen)

with open(fid_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow([_fid])

