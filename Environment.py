import pysc2
from utils import GameState

# TODO


class BicNetAgent(pysc2.agents.base_agent.BaseAgent):

    def __init__(self):
        '''
        init instance
        '''
        super(BicNetAgent, self).__init__()

    def step(self, obs):
        '''
        input: obs
        output: action
        '''
        pass

# TODO


class Environment:

    def __init__(self, screen_size, minimap_size):
        '''
        init environment instance
        '''
        self.screen_size = screen_size
        self.minimap_size = minimap_size

    def init_env(self):
        '''
        init the running environment
        only do once
        '''
        pass

    def reset_env(self):
        '''
        reset the environment between episode
        '''
        pass

    def get_current_state(self) -> GameState:
        '''
        return current state information
        '''
        pass

    def transit(self, action):
        '''
        transit from current state to next using action
        '''
        pass

    def get_reward(self):
        '''
        get the reward of last action
        '''
        pass
