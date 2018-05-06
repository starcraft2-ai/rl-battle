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

    def build_model(self):
        pass

    def set_model(self, model):
        pass

    def load_model(self, checkpoint_dir):
        pass

    def save_model(self, checkpoint_dir):
        pass


class A2CEnvironment(Environment):
    def __init__(self, lock: Lock, agent_class, model=None, M=16, gamma=0.2):
        '''
        init environment instance
        '''
        self.lock = lock
        self.agent : ModelAgent = agent_class()
        self.model = model
        self.agent.model = model
        self.replay_buffer = []
        self.last_observation = None
        self.last_action = None
        self.last_value = None
        self.last_coordinate = None
        self.M = M
        self.gamma = gamma
        self.action_spec = None
        self.observation_spec = None

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
        '''
        # take long CPU time
        with tfe.GradientTape() as tape:
            result = self.agent.step(observation)

            action     = self.agent.last_action
            value      = self.agent.last_value
            coordinate = self.agent.last_coordinate

            if self.last_observation is not None:
                '''
                Actor-Critic Algorithm Here
                This is actaully transition for last round
                because we are using passive mode of game environemnt for actions
                '''
                transition = (self.last_observation, self.last_action, observation)
                last_reward = self.last_observation.reward
                self.replay_buffer.append(transition)

                # TODO: discover the possibliby of using batch on PySC2
                # Maybe not
                # samples = random.sample(self.replay_buffer, self.M)
                # for (observation_0, action, observation_1) in samples:
                #     (_, _, value_1) = self.agent.simulate(observation_1)

                # Online
            
                target_value = last_reward + self.gamma * value

                #TODO: why
                advantage = tf.stop_gradient(target_value - self.last_value)

                coordinate_log_prob = tf.log(
                    tf.clip_by_value(
                        tf.reduce_sum(self.last_coordinate)
                        , 1e-10, 1.
                ))
                action_log_prob = tf.log(
                    tf.clip_by_value(
                        tf.reduce_sum(self.last_action)
                        , 1e-10, 1.
                ))

                loss_policy = - tf.reduce_mean((coordinate_log_prob + action_log_prob) * advantage)
                loss_value = - tf.reduce_mean(target_value * advantage)

                loss = loss_policy + loss_value

                # TODO: figureout a way to bp on action
                #
                # loss = A * log_loss(grad last_action)

                tf.contrib.summary.scalar('loss', loss)

                # Tape
                grads = tape.gradient(loss, self.agent.model.variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.agent.model.variables))

        self.last_observation = observation
        self.last_action      = action
        self.last_value       = value
        self.last_coordinate  = coordinate
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

    def get_optimizer_and_node(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        root = tfe.Checkpoint(optimizer=optimizer,
                                   model=self.model,
                                   optimizer_step=tf.train.get_or_create_global_step())
        return (optimizer, root)

    def set_model(self, model, optimizer, root):
        self.model = model
        self.agent.model = model
        self.optimizer = optimizer
        self.root = root

    def load_model(self, checkpoint_dir):
        if(self.root is None):
            (_, self.root) = self.get_optimizer_and_node()
        self.root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def save_model(self, checkpoint_dir):
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.root.save(file_prefix=checkpoint_prefix)
