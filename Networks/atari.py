import tensorflow as tf
import tensorflow.contrib.eager as tfe

class AtariModel(tf.keras.Model):
  def __init__(self, ssize, msize, num_action):
    super(AtariModel, self).__init__()
    self.ssize = ssize
    self.msize = msize
    self.num_action = num_action
    print('screen_size:', self.ssize)
    print('minimap_size:', self.msize)
    print('num action:', self.num_action)

    self.mconv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, data_format='channels_first')
    self.mconv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2)
    self.mconv_flatten = tf.keras.layers.Flatten()
    self.sconv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4)
    self.sconv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2)
    self.sconv_flatten = tf.keras.layers.Flatten()
    self.aa_flatten = tf.keras.layers.Flatten()
    self.aa_fc = tf.keras.layers.Dense(units=256, activation=tf.nn.tanh)
    self.feat_fc = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
    self.x_fc = tf.keras.layers.Dense(units=ssize, activation=tf.nn.softmax)
    self.y_fc = tf.keras.layers.Dense(units=ssize, activation=tf.nn.softmax)
    self.coordinate_flatten = tf.keras.layers.Flatten()
    self.action_fc = tf.keras.layers.Dense(units=num_action, activation=tf.nn.softmax)
    self.value_fc = tf.keras.layers.Dense(units=1, activation=None)

  def call(self, inputs):
    # extract inputs
    minimap, screen, available_actions = inputs

    # handle minimap conv layer
    # minimap = tf.transpose(minimap, [0, 2, 3, 1])
    print('eval minimap shape', minimap.shape)
    mconv = self.mconv1(minimap)
    mconv = self.mconv2(mconv)

    # handle screen conv layer
    screen = tf.transpose(screen, [0, 2, 3, 1])
    sconv = self.sconv1(screen)
    sconv = self.sconv2(sconv)

    # handle information -  available actions
    available_actions = self.aa_flatten(available_actions)
    aa_fc = self.aa_fc(available_actions)

    # concatenate and connect to a fc layer
    feat_fc = tf.concat([self.mconv_flatten(mconv), self.sconv_flatten(sconv), aa_fc], axis=1)
    feat_fc = self.feat_fc(feat_fc)

    # generate spatial information
    x_fc, y_fc = self.x_fc(feat_fc), self.y_fc(feat_fc)
    x_fc = tf.reshape(x_fc, [-1, 1, self.ssize])
    x_fc = tf.tile(x_fc, [1, self.ssize, 1])
    y_fc = tf.reshape(y_fc, [-1, self.ssize, 1])
    y_fc = tf.tile(y_fc, [1, 1, self.ssize])
    coordinate = self.coordinate_flatten(x_fc * y_fc)

    # generate the action to be taken
    action = self.action_fc(feat_fc)

    # generate the value
    value = self.value_fc(feat_fc)
    value = tf.reshape(value, [-1])
    return coordinate, action, value

  def predict(self, inputs):
    return self.call(inputs)
  
  def temp_call(self, inputs):
    # extract inputs
    (minimap, screen, available_actions) = inputs

    # handle minimap conv layer
    # minimap = tf.transpose(minimap, [0, 2, 3, 1])
    # print('shape', minimap.shape)
    # print('conv layer input', self.mconv1)
    print('train minimap shape', minimap.shape)
    mconv = self.mconv1(minimap)
    # mconv = self.mconv2(mconv)

    # handle screen conv layer
    # screen = tf.transpose(screen, [0, 2, 3, 1])
    # sconv = self.sconv1(screen)
    # sconv = self.sconv2(sconv)

    # # handle information -  available actions
    # available_actions = self.aa_flatten(available_actions)
    # aa_fc = self.aa_fc(available_actions)

    # concatenate and connect to a fc layer
    # feat_fc = tf.concat([self.mconv_flatten(mconv), self.sconv_flatten(sconv), aa_fc], axis=1)
    # feat_fc = self.feat_fc(feat_fc)

    # generate spatial information
    # x_fc, y_fc = self.x_fc(feat_fc), self.y_fc(feat_fc)
    # x_fc = tf.reshape(x_fc, [-1, 1, self.ssize])
    # x_fc = tf.tile(x_fc, [1, self.ssize, 1])
    # y_fc = tf.reshape(y_fc, [-1, self.ssize, 1])
    # y_fc = tf.tile(y_fc, [1, 1, self.ssize])
    # coordinate = self.coordinate_flatten(x_fc * y_fc)

    # generate the action to be taken
    # print('fc layer ', self.action_fc)
    # action = self.action_fc(feat_fc)

    # generate the value
    # value = self.value_fc(feat_fc)
    # value = tf.reshape(value, [-1])
    # return coordinate, action, value

if __name__=='__main__':
    tf.enable_eager_execution()
    ssize, msize = 64, 64
    num_action = 10
    batch_size = 6
    schannel, mchannel = 8, 8
    model = AtariModel(ssize=ssize, msize=msize, num_action=num_action)
    inputs = tf.random_normal([batch_size,mchannel,msize,msize]), tf.random_normal([batch_size,schannel,ssize,ssize]), tf.random_normal([batch_size, num_action])
    print(model.predict(inputs))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
