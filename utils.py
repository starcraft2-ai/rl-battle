from typing import Tuple, List
import tensorflow as tf

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
