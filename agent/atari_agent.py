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
    def __init__(self, name='AtariAgent', model=None):
        super().__init__()
        self.name = name
        self.model: AtariModel = model
        self.rewards = [0]

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        self.build_model()

    def reset(self):
        super().reset()

    def simulate(self, obs):
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
        return (coordinate, action, value)

    def step(self, obs):
        super().step(obs)
        self.rewards[-1] += obs.reward
        if obs.last():
            self.rewards.append(0)

        # predict
        (coordinate, action, value) = self.simulate(obs)

        # reduce dimentsion
        temp = tf.argmax(coordinate, 1)[0]
        y, x = temp // self.obs_spec['screen'][0], temp % self.obs_spec['screen'][0]
        #print('x:{x}, y:{y}'.format(x=x,y=y))
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

        # set value for public access or train use
        self.last_value = value
        self.last_action = (action_selected, act_args)

        return actions.FunctionCall(action_selected, act_args)

    def build_model(self, initializer=tf.zeros):
        self.model = AtariModel(
            self.obs_spec["screen"][0], self.obs_spec["minimap"][0], possible_action_num)

        return self.model
