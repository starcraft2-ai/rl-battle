from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions
from tensorflow.contrib import eager as tfe
tfe.enable_eager_execution()

class AtariAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        
    def reset(self):
        super().reset()

    def step(self, obs):
        super().step(obs)
        (screen, minimap) = (
            tfe.Variable(obs.observation['screen']),
            tfe.Variable(obs.observation['minimap'])
        )

    def build_model(self):
        pass

    def load_model(self):
        pass
    
    def save_model(self):
        pass
