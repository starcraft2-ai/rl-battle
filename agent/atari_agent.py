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
        self.obs_spec = None
        self.action_spec = None

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

    def reset(self):
        super().reset()

    def fit(self, obs):
        (screen, minimap, available_actions) = (
            tf.constant(obs.observation['screen'], tf.float32),
            tf.constant(obs.observation['minimap'], tf.float32),
            np.zeros([possible_action_num], dtype=np.float32)
        )
        available_actions[obs.observation['available_actions']] = 1

        # induce dimension
        input = (
            tf.expand_dims(minimap, 0),
            tf.expand_dims(screen, 0),
            tf.expand_dims(available_actions, 0)
        )

        # predict
        return self.model.predict(input)

    def step(self, obs):
        super().step(obs)
        self.rewards[-1] += obs.reward
        if obs.last():
            self.rewards.append(0)

        # predict 
        # -- same as fit
        (screen, minimap, available_actions) = (
            tf.constant(obs.observation['screen'], tf.float32),
            tf.constant(obs.observation['minimap'], tf.float32),
            np.zeros([possible_action_num], dtype=np.float32)
        )
        available_actions[obs.observation['available_actions']] = 1

        # induce dimension
        input = (
            tf.expand_dims(minimap, 0),
            tf.expand_dims(screen, 0),
            tf.expand_dims(available_actions, 0)
        )

        # predict
        (coordinate, action, value) = self.model.predict(input)
        self.last_value = value[0]
        self.last_action = action[0]
        self.last_coordinate = coordinate[0]

        # -- same as fit -- 

        # reduce dimentsion
        y, x = (
            tf.argmax(coordinate, 1)[0].numpy() // self.obs_spec['screen'][1], 
            tf.argmax(coordinate, 1)[0].numpy() %  self.obs_spec['screen'][1]
        )
        action = action[0]
        value = value[0]

        # select available_actions
        action_selected = tf.argmax(action * available_actions).numpy()

        # form action and call
        # TODO: better implementation
        act_args = []
        for arg in actions.FUNCTIONS[action_selected].args:
            if arg.name in ('screen', 'screen2', 'minimap'):
                act_args.append([x, y])
            else:
                act_args.append([0])

        # set value for public access or train use
        

        print(actions.FUNCTIONS[action_selected].args)
        return actions.FunctionCall(action_selected, act_args)

    def build_model(self, initializer=tf.zeros):
        self.model = AtariModel(
            self.obs_spec["screen"][1], self.obs_spec["minimap"][1], possible_action_num)
        return self.model
