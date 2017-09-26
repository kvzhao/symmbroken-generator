from __future__ import division
from __future__ import print_function

import os, sys
import collections
import numpy as np
import tensorflow as tf

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tflib as lib
import tflib.ops.linear
from data import brokenstate, cal_mag

# model
tf.app.flags.DEFINE_integer('model_dim', 128, 'Dimensionality of the model')

# training 
tf.app.flags.DEFINE_integer('num_steps', 10000, 'Number steps of traning')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('critic_steps', 5, 'Number of updates of critic network')

# dataset
tf.app.flags.DEFINE_float('deviation', 0.01, 'Standard deviation of ising state')
tf.app.flags.DEFINE_integer('L', 10, 'System size of Ising model')
tf.app.flags.DEFINE_integer('NUM_DATA', 10000, 'Number of broken states data')
tf.app.flags.DEFINE_float('LAMBDA', 0.1, 'lambda is used in gradient penalty')
tf.app.flags.DEFINE_integer('SAVE_FREQ', 200, 'Interval steps of saving results')
tf.app.flags.DEFINE_string('out', 'results', 'Path to output folder')

FLAGS = tf.app.flags.FLAGS

N = FLAGS.L ** 2
print ('Number of sites: {}'.format(N))

if not tf.gfile.IsDirectory(FLAGS.out):
    tf.gfile.MakeDirs(FLAGS.out)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in, n_out, inputs, initialization='he'
    )
    output = tf.nn.relu(output)
    return output

# Check the input dimension

def Generator(n_samples, real_data):
    DIM = FLAGS.model_dim
    noise = tf.random_normal([n_samples, 2])
    output = ReLULayer('Generator.1', 2, DIM, noise)
    output = ReLULayer('Generator.2', DIM, DIM, output)
    output = ReLULayer('Generator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Generator.4', DIM, N, output)
    return output

def Discriminator(inputs):
    DIM = FLAGS.model_dim
    output = ReLULayer('Discriminator.1', N, DIM, inputs)
    output = ReLULayer('Discriminator.2', DIM, DIM, output)
    output = ReLULayer('Discriminator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Discriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[None, N])
fake_data = Generator(FLAGS.batch_size, real_data)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

# WGAN loss
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

alpha = tf.random_uniform(shape=[FLAGS.batch_size, 1], minval=0., maxval=1.)
interpolates = alpha*real_data + ((1-alpha)*fake_data)
disc_interpolates = Discriminator(interpolates)
gradients = tf.gradients(disc_interpolates, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1)**2)

disc_cost += FLAGS.LAMBDA * gradient_penalty

disc_params = lib.params_with_name('Discriminator')
gen_params = lib.params_with_name('Generator')

disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
if len(gen_params) > 0:
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
else:
    gen_train_op = tf.no_op()

# A good way to show the network in command line
print ("Generator params:")
for var in lib.params_with_name('Generator'):
    print ("\t{}\t{}".format(var.name, var.get_shape()))
print ("Discriminator params:")
for var in lib.params_with_name('Discriminator'):
    print ("\t{}\t{}".format(var.name, var.get_shape()))


def save_image(name, spins):
    mags = []
    for s in spins:
        mags.append(cal_mag(s))
    weights = np.ones_like(mags) /float(len(mags))
    plt.hist(mags, bins=100, weights=weights)
    plt.xlabel('magnetization')
    plt.savefig('{}/{}.png'.format(FLAGS.out, name))
    plt.clf()

# Prepare dataset
def gen_data():
    dataset=[]
    for i in range(FLAGS.NUM_DATA):
        s, _ = brokenstate(FLAGS.L, FLAGS.deviation)
        dataset.append(s)
    dataset = np.array(dataset, dtype=np.float32)
    np.random.shuffle(dataset)

    save_image('dataset', dataset)

    # following code i do not know
    while True:
        for i in xrange(int(len(dataset)/FLAGS.batch_size)):
            yield dataset[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]


# TRAINING.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = gen_data()
    for step in range(FLAGS.num_steps):
        # Train Generator
        if step > 0:
            _= sess.run(gen_train_op)
        # Train Critic
        for i in range(FLAGS.critic_steps):
            batch_data = data.next()
            _disc_cost, _ = sess.run([disc_cost, disc_train_op], 
                feed_dict={real_data: batch_data})
        if step % 100 == 0:
            print ('step: {}, critic cost = {}'.format(step, _disc_cost))
        
        if step % FLAGS.SAVE_FREQ == 0:
            # Generate bunch of data (5 times of batch size)
            samples = []
            for _ in range(5):
                samples.extend(sess.run(fake_data))
            name = 'step_' + str(step)
            save_image(name, samples)