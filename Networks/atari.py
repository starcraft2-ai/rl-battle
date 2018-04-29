from typing import Tuple, List

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.eager as tfe

def build_atari(minimap, screen, info, msize: int, ssize: int, num_action: int):
    # Extract features
    mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                           num_outputs=16,
                           kernel_size=8,
                           stride=4,
                           scope='mconv1')
    mconv2 = layers.conv2d(mconv1,
                           num_outputs=32,
                           kernel_size=4,
                           stride=2,
                           scope='mconv2')
    sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                           num_outputs=16,
                           kernel_size=8,
                           stride=4,
                           scope='sconv1')
    sconv2 = layers.conv2d(sconv1,
                           num_outputs=32,
                           kernel_size=4,
                           stride=2,
                           scope='sconv2')
    info_fc = layers.fully_connected(layers.flatten(info),
                                     num_outputs=256,
                                     activation_fn=tf.tanh,
                                     scope='info_fc')

    # Compute spatial actions, non spatial actions and value
    feat_fc = tf.concat(
        [layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
    feat_fc = layers.fully_connected(feat_fc,
                                     num_outputs=256,
                                     activation_fn=tf.nn.relu,
                                     scope='feat_fc')

    spatial_action_x = layers.fully_connected(feat_fc,
                                              num_outputs=ssize,
                                              activation_fn=tf.nn.softmax,
                                              scope='spatial_action_x')
    spatial_action_y = layers.fully_connected(feat_fc,
                                              num_outputs=ssize,
                                              activation_fn=tf.nn.softmax,
                                              scope='spatial_action_y')
    spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
    spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
    spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
    spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
    spatial_action = layers.flatten(spatial_action_x * spatial_action_y)

    non_spatial_action = layers.fully_connected(feat_fc,
                                                num_outputs=num_action,
                                                activation_fn=tf.nn.softmax,
                                                scope='non_spatial_action')
    value = tf.reshape(layers.fully_connected(feat_fc,
                                              num_outputs=1,
                                              activation_fn=None,
                                              scope='value'), [-1])

    return spatial_action, non_spatial_action, value

class AtariModel(tf.keras.Model):
#   def __init__(self):
#     super(Model, self).__init__()
#     self.W = tfe.Variable(5., name='weight')
#     self.B = tfe.Variable(10., name='bias')
  def predict(self, inputs):
    minimap, screen, info, msize, ssize, num_action = inputs
    return build_atari(minimap, screen, info, msize, ssize, num_action)

if __name__=='__main__':
    tf.enable_eager_execution()
    model = AtariModel()
    inputs = tf.zeros([8,8,64,64]), tf.zeros([8,8,64,64]), tf.zeros([8, 10]), 64, 64, 10
    print(model.predict(inputs))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)