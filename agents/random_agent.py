"""A random agent for starcraft."""

from absl import flags

import numpy

import sys

from pysc2.agents import base_agent
from pysc2.lib import actions

class RandomAgent(base_agent.BaseAgent):

    def __init__(self, max_steps = 2500):
        super(RandomAgent, self).__init__()
        self.rewards = [0]
        self.max_steps = max_steps

    """A random agent for starcraft."""

    def step(self, obs):
        super(RandomAgent, self).step(obs)
        self.rewards[-1] += obs.reward
        if obs.last():
            print('episode:{episode}, step:{step}, reward:{reward}'.format(
                episode=self.episodes, step=self.steps, reward=self.rewards[-1]), file=sys.stderr)
            self.rewards.append(0)
        if self.steps == self.max_steps:
            print('rewards:', self.rewards)
            print('mean:', sum(self.rewards) / len(self.rewards))
            print('max:', max(self.rewards))
        function_id = numpy.random.choice(obs.observation["available_actions"])
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)
