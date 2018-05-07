import pysc2
import os
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

    def set_sepcs(self, action_spec, observation_spec):
        pass

    def setup(self, obs_spec, action_spec):
        pass

    def reset(self):
        pass

    def step(self, observation):
        pass

    def after_step(self, before_step_observation, observation):
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
    def __init__(self, lock: Lock, agent_class, model=None, M=16, gamma=0.2, writer = None):
        '''
        init environment instance
        '''
        self.lock = lock
        self.agent : ModelAgent = agent_class()
        self.model = model
        self.agent.model = model
        self.replay_buffer = []
        self.last_features = None
        self.M = M
        self.gamma = gamma
        self.action_spec = None
        self.observation_spec = None
        self.summary_n_step = 100
        self.global_step = None
        self.writer = writer

    def set_sepcs(self, action_spec, observation_spec):
        self.agent.set_sepcs(action_spec, observation_spec)

    def setup(self, obs_spec, action_spec):
        '''
        init the running environment
        only do once
        '''
        assert self.model is not None, 'No model provided to environment. either intialize one or set one!'
        self.agent.setup(obs_spec, action_spec)

    def reset(self):
        '''
        reset the environment between episode
        '''
        self.agent.reset()

    def step(self, observation):
        '''
        transit from current state to next using action
        actaully `observation` is `last_observation`
        '''
        # take long CPU time
        if(self.last_features):
            result = self.agent.act(self.last_features)
        else:
            result = self.agent.step(observation)
           
        return result

    def after_step(self, before_step_observation, observation):
        '''
        Actor-Critic Algorithm
        '''
        # transition = (before_step_observation, self.last_action, observation)
        # last_reward = self.last_observation.reward
        # self.replay_buffer.append(transition)

        # TODO: discover the possibliby of using batch on PySC2
        # Maybe not
        # samples = random.sample(self.replay_buffer, self.M)
        # for (observation_0, action, observation_1) in samples:
        #     (_, _, value_1) = self.agent.simulate(observation_1)

        # Online
        self.global_step.assign_add(1, use_locking=True)
        with tfe.GradientTape() as tape:
            (
                before_coordinate, 
                before_action, 
                before_value
            ) = self.agent.fit(before_step_observation)
            (coordinate, action, value) = self.agent.fit(observation)
            self.last_features = (coordinate, action, value)

            target_value = before_step_observation.reward + self.gamma * value[0]

            #TODO: why
            advantage = tf.stop_gradient(target_value - before_value)

            coordinate_log_prob = tf.log(
                tf.clip_by_value(
                    tf.reduce_sum(before_coordinate)
                    , 1e-10, 1.
            ))
            action_log_prob = tf.log(
                tf.clip_by_value(
                    tf.reduce_sum(before_action)
                    , 1e-10, 1.
            ))

            loss_policy = - tf.reduce_mean((coordinate_log_prob + action_log_prob) * advantage)
            loss_value = - tf.reduce_mean(target_value * advantage)

            loss = loss_policy + loss_value

            if self.writer:
                with self.lock and self.writer.as_default():
                    with tf.contrib.summary.record_summaries_every_n_global_steps(self.summary_n_step):
                        tf.contrib.summary.scalar('score', self.agent.reward)
                        tf.contrib.summary.scalar('loss/policy', loss_policy)
                        tf.contrib.summary.scalar('loss/value', loss_value)
                        tf.contrib.summary.scalar('loss', loss)

        # Tape
        grads = tape.gradient(loss, self.agent.model.variables)
        self.optimizer.apply_gradients(
            zip(grads, self.agent.model.variables))

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

    def get_optimizer_and_node(self):
        return self.agent.get_optimizer_and_node()

    def set_model(self, model, optimizer, root):
        self.model = model
        self.agent.model = model
        self.optimizer = optimizer
        self.root = root
        self.global_step = tf.train.get_or_create_global_step()

    def load_model(self, checkpoint_dir):
        self.agent.load_model(checkpoint_dir)

    def save_model(self, checkpoint_dir):
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        with self.lock:
            self.root.save(file_prefix=checkpoint_prefix)
