from typing import Tuple, List

GameState = Tuple[List[float], List[float]]
Actions = List[Tuple[int, float]]
Transition = Tuple[GameState, Actions, GameState]

import random

class Buffer:
    def __init__(self, size = 2500):
        """
        dequeued when filled with size
        """
        self.max_size = size
        self.queue : List[Transition] = []

    def add(self, transiton : Transition):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(transiton)

    def sample(self, size: int = 10) -> List[Transition]:
        """
        Fetch random size 
        """
        return random.sample(self.queue, size)
