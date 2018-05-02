import pysc2
from utils import GameState
from agent.model_agent_protocal import ModelAgent
from multiprocessing import Lock

import tensorflow as tf
from tensorflow.contrib import eager as tfe

import random

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
    def __init__(self, lock: Lock, agent_class: ModelAgent.__class__, model=None, M = 16, gamma = 0.2):
        '''
        init environment instance
        '''
        self.lock = lock
        self.agent = agent_class()
        self.model = model
        self.agent.model = model
        self.replay_buffer = []
        self.last_observation = None
        self.last_action = None
        self.last_value = None
        self.M = M
        self.gamma = gamma

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
        result =  self.agent.step(observation)

        action      = self.agent.last_action
        value       = self.agent.last_value

        if self.last_observation is not None:
            '''
            Actor-Critic Algorithm Here
            This is actaully transition for last round
            because we are using passive mode of game environemnt for actions
            '''
            transition = (self.last_observation, self.last_action , observation)
            last_reward = self.last_observation.reward
            self.replay_buffer.append(transition)

            # TODO: discover the possibliby of using batch on PySC2
            # Maybe not
            # samples = random.sample(self.replay_buffer, self.M)
            # for (observation_0, action, observation_1) in samples:
            #     (_, _, value_1) = self.agent.simulate(observation_1)

            # Online          
            with tf.GradientTape() as tape:
                target_y = last_reward + self.gamma * value
                A = target_y - self.last_value 
                # value_loss = tf.squared_difference(self.last_value,  target_y)
                
                # TODO: figureout a way to bp on action
                # 
                # loss = A * log_loss(grad last_action)


                tf.contrib.summary.scalar('loss', A)
            grads = tape.gradient(A, self.agent.model.variables)
            self.optimizer.apply_gradients(zip(grads, self.agent.model.variables))

        self.last_observation = observation
        self.last_action      = action
        self.last_value       = value
        return result

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
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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
