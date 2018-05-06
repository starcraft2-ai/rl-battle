# from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions

class ModelAgent(object):
    def __init__(self):
        self.reward = 0
        self.rewards = [0]
        self.episodes = 0
        self.steps = 0
        self.last_value = 0

    def set_sepcs(self, action_spec, observation_spec):
        self.obs_spec = observation_spec
        self.action_spec = action_spec

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1

    def fit(self, obs):
        pass

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward
        # episodic
        self.rewards[-1] += obs.reward
        if obs.last():
            self.rewards.append(0)
        return actions.FunctionCall(0, [])

    def act(self, features):
        """
        Don't call this if you've called step already
        """
        return actions.FunctionCall(0, [])

    def build_model(self, initializer):
        pass
