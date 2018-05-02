import pysc2
from utils import GameState
from agent.model_agent_protocal import ModelAgent
from multiprocessing import Lock

import tensorflow as tf
from tensorflow.contrib import eager as tfe


class Environment:
    def __init__(self, lock: Lock, agent_class: ModelAgent.__class__, model=None):
        self.lock = lock
        self.agent = agent_class()
        self.model = model
        self.agent.model = model
        raise Exception(
            "Should not instentiate Environment, use A2CEnvironment instead!")

    def setup(self, obs_spec, action_spec):
        pass

    def reset(self):
        pass

    def step(self, observation):
        pass

    def build_model(self):
        pass

    def set_model(self, model):
        pass

    def load_model(self, checkpoint_dir):
        pass

    def save_model(self, checkpoint_dir):
        pass


class A2CEnvironment(Environment):
    def __init__(self, lock: Lock, agent_class: ModelAgent.__class__, model=None):
        '''
        init environment instance
        '''
        self.lock = lock
        self.agent = agent_class()
        self.model = model
        self.agent.model = model
        self.replay_buffer = []

    def setup(self, obs_spec, action_spec):
        '''
        init the running environment
        only do once
        '''
        assert self.model is not None, 'No model provided to environment. either intialize one or set one!'
        self.agent.step(obs_spec, action_spec)

    def reset(self):
        '''
        reset the environment between episode
        '''
        self.agent.reset()

    def step(self, observation):
        '''
        transit from current state to next using action
        '''
        # take long CPU time
        action = self.agent.step(observation)
        #
        reward = observation.reward
        state = observation.observation
        value = self.agent.last_value

        loss = reward

        return action

    def set_replay_buffer(self, replay_buffer):
        '''
        Use if you want to share replay buffer with other instences
        '''
        self.replay_buffer = replay_buffer

    def build_model(self):
        '''
        Build and get Agent Model
        '''
        self.model = self.agent.build_model()
        return self.model

    def init_root(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.root = tfe.Checkpoint(optimizer=optimizer,
                                   model=self.model,
                                   optimizer_step=tf.train.get_or_create_global_step())

    def set_model(self, model):
        self.model = model
        self.agent.model = model

    def load_model(self, checkpoint_dir):
        self.root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def save_model(self, checkpoint_dir):
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.root.save(file_prefix=checkpoint_prefix)
