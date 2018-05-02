# from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions

class ModelAgent(object):
    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.last_value = 0

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1

    def simulate(self, obs):
        pass

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward
        return actions.FunctionCall(0, [])

    def build_model(self, initializer):
        pass
