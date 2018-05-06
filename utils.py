from typing import Tuple, List
import tensorflow as tf
from pysc2.lib import actions
import numpy as np
from copy import deepcopy

CommonState = List[float]
AgentState = List[float]
GameState = Tuple[CommonState, AgentState]
Coordinates = Tuple[float, float]
ActionProbablity = List[float]
ActionTable = List[ActionProbablity]
Actions = List[int]
Transition = Tuple[GameState, Actions, GameState]

possible_action_num = len(actions.FUNCTIONS)

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
    episode_rb = deepcopy(episode_rb)

    # Compute targets and masks
    minimaps = []
    screens = []
    available_actions = []

    episode_rb.reverse()
    for _, [obs, _, _] in enumerate(episode_rb):
        minimap, screen, available_action = model_input(obs)

        minimaps.append(minimap.numpy())
        screens.append(screen.numpy())
        available_actions.append(available_action.numpy())

    minimaps = tf.constant(minimaps, tf.float32)
    screens = tf.constant(screens, tf.float32)
    available_actions = tf.constant(available_actions, tf.float32)
    return minimaps, screens, available_actions

# collect coordinate-related feature
def collect_coordinate_feature(episode_rb, ssize):
    # deepcopy the array, where element is not deepcopied, but that does not matter
    episode_rb = deepcopy(episode_rb)

    valid_coordinate = np.zeros([len(episode_rb)], dtype=np.float32)
    selected_coordinate = np.zeros([len(episode_rb), ssize ** 2], dtype=np.float32)

    episode_rb.reverse()
    for i, [_, action, _] in enumerate(episode_rb):
        act_id = action.function
        act_args = action.arguments

        args = actions.FUNCTIONS[act_id].args
        for arg, act_arg in zip(args, act_args):
            if arg.name in ('screen', 'minimap', 'screen2'):
                ind = act_arg[1] * ssize + act_arg[0]
                valid_coordinate[i] = 1
                selected_coordinate[i, ind] = 1
    return valid_coordinate, selected_coordinate

def collect_action_feature(episode_rb):
    # deepcopy the array, where element is not deepcopied, but that does not matter
    episode_rb = deepcopy(episode_rb)

    valid_action = np.zeros([len(episode_rb), possible_action_num], dtype=np.float32)
    selected_action = np.zeros([len(episode_rb), possible_action_num], dtype=np.float32)

    episode_rb.reverse()
    for i, [obs, action, _] in enumerate(episode_rb):
        act_id = action.function
        valid_action[i, obs.observation["available_actions"]] = 1
        selected_action[i, act_id] = 1
    return valid_action, selected_action