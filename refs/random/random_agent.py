# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import numpy

import sys

from pysc2.agents import base_agent
from pysc2.lib import actions

FLAGS = flags.FLAGS

print('Max steps:{max_step}'.format(max_step=FLAGS.max_agent_steps), file=sys.stderr)

class RandomAgent(base_agent.BaseAgent):

  def __init__(self):
    super(RandomAgent, self).__init__()
    self.rewards = [0]

  """A random agent for starcraft."""

  def step(self, obs):
    super(RandomAgent, self).step(obs)
    self.rewards[-1] += obs.reward
    if obs.last():
      print('episode:{episode}, step:{step}, reward:{reward}'.format(episode=self.episodes,step=self.steps, reward=self.rewards[-1]), file=sys.stderr)
      self.rewards.append(0)
    if self.steps == FLAGS.max_agent_steps:
      print('rewards:', self.rewards)
      print('mean:', sum(self.rewards) / len(self.rewards))
      print('max:', max(self.rewards))
    function_id = numpy.random.choice(obs.observation["available_actions"])
    args = [[numpy.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]
    return actions.FunctionCall(function_id, args)
