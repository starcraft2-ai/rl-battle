import os
from agent.model_agent_protocal import ModelAgent
from pysc2.lib import actions
import tensorflow as tf
from tensorflow.contrib import eager as tfe
import numpy as np
tfe.enable_eager_execution()
from Networks.atari import AtariModel


possible_action_num = len(actions.FUNCTIONS)


class AtariAgent(ModelAgent):
    def __init__(self, name='AtariAgent'):
        super().__init__()
        self.name = name
        self.model: AtariModel = None
        self.rewards = [0]

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        self.build_model()

    def reset(self):
        super().reset()

    def step(self, obs):
        super().step(obs)
        self.rewards[-1] += obs.reward
        if obs.last():
            self.rewards.append(0)
        (screen, minimap, available_actions) = (
            tf.constant(obs.observation['screen'], tf.float32),
            tf.constant(obs.observation['minimap'], tf.float32),
            np.zeros([possible_action_num], dtype=np.float32)
        )
        available_actions[obs.observation['available_actions']] = 1

        # induce dimension
        x = (
            tf.expand_dims(minimap, 0),
            tf.expand_dims(screen, 0),
            tf.expand_dims(available_actions, 0)
        )

        # predict
        (coordinate, action, value) = self.model.predict(x)

        # reduce dimentsion
        y, x = coordinate
        y, x = y[0], x[0]
        action = action[0]
        value = value[0]

        # select available_actions
        action_selected = tf.argmax(action * available_actions).numpy()

        # form action and call
        # TODO: better implementation
        act_args = []
        for arg in actions.FUNCTIONS[action_selected].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([x, y])
            else:
                act_args.append([0])
        return actions.FunctionCall(action_selected, act_args)

    def build_model(self, initializer=tf.zeros):
        self.model = AtariModel(
            self.obs_spec["screen"][0], self.obs_spec["minimap"][0], possible_action_num)

        # TODO: Training
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.root = tfe.Checkpoint(optimizer=optimizer,
                                   model=self.model,
                                   optimizer_step=tf.train.get_or_create_global_step())

    def load_model(self, checkpoint_dir):
        self.root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def save_model(self, checkpoint_dir):
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.root.save(file_prefix=checkpoint_prefix)

    #TODO
    def loss(coordinate, action, value, valid_coordinate, selected_coordinate, valid_action, selected_action, target_value):
      # Compute log probability
      coordinate_prob = tf.reduce_sum(coordinate * selected_coordinate, axis=1)
      coordinate_log_prob = tf.log(tf.clip_by_value(coordinate_prob, 1e-10, 1.))
      action_prob = tf.reduce_sum(action * selected_action, axis=1)
      valid_action_prob = tf.reduce_sum(action * valid_action, axis=1)
      valid_action_prob = tf.clip_by_value(valid_action_prob, 1e-10, 1.)
      action_prob = action_prob / valid_action_prob
      action_log_prob = tf.log(tf.clip_by_value(action_prob, 1e-10, 1.))
      # TODO
      # record action_prob and coordinate_prob
      # self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
      # self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

      # Compute losses, more details in https://arxiv.org/abs/1602.01783
      # Policy loss and value loss
      action_log_prob = valid_coordinate * coordinate_log_prob + action_log_prob
      error = tf.stop_gradient(target_value - value)
      policy_loss = - tf.reduce_mean(action_log_prob * error)
      value_loss = - tf.reduce_mean(self.value * error)
      #TODO
      # self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
      # self.summary.append(tf.summary.scalar('value_loss', value_loss))
      return policy_loss + value_loss

    #TODO
    def train_model():
      pass
