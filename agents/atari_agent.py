from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions

class AtariAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        
    def reset(self):
        super().reset()

    def step(self, obs):
        super().step(obs)

    def build_model(self):
        pass

    def load_model(self):
        pass
    
    def save_model(self):
        pass
