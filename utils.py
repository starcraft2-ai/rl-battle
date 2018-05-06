from typing import Tuple, List
import tensorflow as tf
from pysc2.lib import actions
import numpy as np

CommonState = List[float]
AgentState = List[float]
GameState = Tuple[CommonState, AgentState]
Coordinates = Tuple[float, float]
ActionProbablity = List[float]
ActionTable = List[ActionProbablity]
Actions = List[int]
Transition = Tuple[GameState, Actions, GameState]

import random


class Buffer:
    def __init__(self, size=2500):
        """
        dequeued when filled with size
        """
        self.max_size = size
        self.queue: List[Transition] = []

    def add(self, transiton: Transition):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(transiton)

    def sample(self, size: int = 10) -> List[Transition]:
        """
        Fetch random size 
        """
        return random.sample(self.queue, size)


def best_actions(action_table: ActionTable) -> Actions:
    return [tf.argmax(action_probablity) for action_probablity in action_table]


# extract minimap, screen, available_actions
def model_input(obs):
    possible_action_num = len(actions.FUNCTIONS)

    (minimap, screen, available_actions) = (
        tf.constant(obs.observation['minimap'], tf.float32),
        tf.constant(obs.observation['screen'], tf.float32),
        np.zeros([possible_action_num], dtype=np.float32)
    )
    available_actions[obs.observation['available_actions']] = 1
    available_actions = tf.constant(available_actions, tf.float32)
    return (minimap, screen, available_actions)

# Compute target value
def calculate_target_value(episode_rb, discount, model):
    # deepcopy the array, where element is not deepcopied, but that does not matter
    from copy import deepcopy
    episode_rb = deepcopy(episode_rb)

    # get last value from last observation
    obs = episode_rb[-1][-1]
    if obs.last():
        R = 0
    else:
        (minimap, screen, available_action) = model_input(obs)

        # induce dimension
        x = (
            tf.expand_dims(minimap, 0),
            tf.expand_dims(screen, 0),
            tf.expand_dims(available_action, 0)
        )
        _, _, R = model(x)
        R = R[0].numpy()

    # Compute targets and masks
    target_value = np.zeros([len(episode_rb)], dtype=np.float32)
    target_value[-1] = R

    episode_rb.reverse()
    for i, [obs, _, _] in enumerate(episode_rb):
        reward = obs.reward
        target_value[i] = reward + discount * target_value[i-1]
    return target_value

# collect model inputs for a single episode
def collect_episode_model_input(episode_rb):
    # deepcopy the array, where element is not deepcopied, but that does not matter
    from copy import deepcopy
    episode_rb = deepcopy(episode_rb)

    # Compute targets and masks
    minimaps = []
    screens = []
    available_actions = []

    episode_rb.reverse()
    for i, [obs, action, _] in enumerate(episode_rb):
        minimap, screen, available_action = model_input(obs)

        minimaps.append(minimap.numpy())
        screens.append(screen.numpy())
        available_actions.append(available_action.numpy())

    minimaps = tf.constant(minimaps, tf.float32)
    screens = tf.constant(screens, tf.float32)
    available_actions = tf.constant(available_actions, tf.float32)
    return minimaps, screens, available_actions

def collect_coordinate_feature(episode_rb):
    pass