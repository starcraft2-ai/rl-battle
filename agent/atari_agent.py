import os
from agent.model_agent_protocal import ModelAgent
from pysc2.lib import actions
import tensorflow as tf
from tensorflow.contrib import eager as tfe
import numpy as np
tfe.enable_eager_execution()
from Networks.atari import AtariModel
import time


possible_action_num = len(actions.FUNCTIONS)

def model_input(obs):
    (screen, minimap, available_actions) = (
        tf.constant(obs.observation['screen'], tf.float32),
        tf.constant(obs.observation['minimap'], tf.float32),
        np.zeros([possible_action_num], dtype=np.float32)
    )
    available_actions[obs.observation['available_actions']] = 1
    return screen, minimap, available_actions

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
        temp = tf.argmax(coordinate, 1)[0]
        y, x = temp // self.obs_spec['screen'][0], temp % self.obs_spec['screen'][0]
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

    def loss(self, coordinate, action, value, valid_coordinate, selected_coordinate, valid_action, selected_action, target_value):
      # Compute log probability
      coordinate_prob = tf.reduce_sum(coordinate * selected_coordinate, axis=1)
      coordinate_log_prob = tf.log(tf.clip_by_value(coordinate_prob, 1e-10, 1.))
      action_prob = tf.reduce_sum(action * selected_action, axis=1)
      valid_action_prob = tf.reduce_sum(action * valid_action, axis=1)
      valid_action_prob = tf.clip_by_value(valid_action_prob, 1e-10, 1.)
      action_prob = action_prob / valid_action_prob
      action_log_prob = tf.log(tf.clip_by_value(action_prob, 1e-10, 1.))

      # Compute losses, more details in https://arxiv.org/abs/1602.01783
      # Policy loss and value loss
      action_log_prob = valid_coordinate * coordinate_log_prob + action_log_prob
      error = tf.stop_gradient(target_value - value)
      policy_loss = - tf.reduce_mean(action_log_prob * error)
      value_loss = - tf.reduce_mean(self.value * error)

      # record losses
      tf.contrib.summary.scalar('policy_loss', policy_loss)
      tf.contrib.summary.scalar('value_loss', value_loss)
      return policy_loss + value_loss

    #TODO
    def train_model(self, optimizer, episode_rb, step_counter, discount, log_interval=None):
        # compute R
        obs = episode_rb[-1][-1]
        if obs.last():
            R = 0
        else:
            minimap, screen, available_action = model_input(obs)

            # induce dimension
            x = (
                tf.expand_dims(minimap, 0),
                tf.expand_dims(screen, 0),
                tf.expand_dims(available_action, 0)
            )
            # TODO: training=True or not?
            _, _, R = self.model(x)

        # Compute targets and masks
        minimaps = []
        screens = []
        available_actions = []

        target_value = np.zeros([len(episode_rb)], dtype=np.float32)
        target_value[-1] = R

        valid_coordinate = np.zeros([len(episode_rb)], dtype=np.float32)
        selected_coordinate = np.zeros([len(episode_rb), self.ssize**2], dtype=np.float32)
        valid_action = np.zeros([len(episode_rb), len(actions.FUNCTIONS)], dtype=np.float32)
        selected_action = np.zeros([len(episode_rb), len(actions.FUNCTIONS)], dtype=np.float32)

        episode_rb.reverse()
        for i, [obs, action, _] in enumerate(episode_rb):
            minimap, screen, available_action = model_input(obs)

            minimaps.append(minimap)
            screens.append(screen)
            available_actions.append(available_action)

            reward = obs.reward
            act_id = action.function
            act_args = action.arguments

            target_value[i] = reward + discount * target_value[i-1]

            valid_action[i, obs.observation["available_actions"]] = 1
            selected_action[i, act_id] = 1

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.obs_spec["screen"][0] + act_arg[0]
                    valid_coordinate[i] = 1
                    selected_coordinate[i, ind] = 1

        minimaps = tf.constant(minimaps, tf.float32)
        screens = tf.constant(screens, tf.float32)
        available_actions = tf.constant(available_actions, tf.float32)
      
        # real training part
        start = time.time()
        x = minimaps, screens, available_actions
        with tf.contrib.summary.record_summaries_every_n_global_steps(10, global_step=step_counter):
            with tfe.GradientTape() as tape:
                coordinate, action, value = self.model(x, training=True)
                loss_value = self.loss(coordinate, action, value, valid_coordinate, selected_coordinate, valid_action, selected_action, target_value)
                tf.contrib.summary.scalar('loss', loss_value)
            grads = tape.gradient(loss_value, self.model.variables)
            optimizer.apply_gradients(zip(grads, self.model.variables), global_step=step_counter)
            if log_interval and step_counter % log_interval == 0:
                rate = log_interval / (time.time() - start)
                print('Step #%d\tLoss: %.6f (%d steps/sec)' % (step_counter, loss_value, rate))

    #   grads = opt.compute_gradients(loss)
    #   cliped_grad = []
    #   for grad, var in grads:
    #     self.summary.append(tf.summary.histogram(var.op.name, var))
    #     self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
    #     grad = tf.clip_by_norm(grad, 10.0)
    #     cliped_grad.append([grad, var])
    #   self.train_op = opt.apply_gradients(cliped_grad)
    #   self.summary_op = tf.summary.merge(self.summary)

    #   self.saver = tf.train.Saver(max_to_keep=100)
    # Compute R, which is value of the last observation