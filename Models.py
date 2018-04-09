from utils import GameState, AgentState, CommonState, ActionTable, Coordinates

# TODO
class BaseNetwork:

    '''
        Base network
    '''

    def __init__(self, **kwargs):
        '''
            init actor network
            input inclues but not only:
            * state size 
            * number of actions
            * input size of BiRNN
            * hidden layer size of BiRNN
            * output size of BiRNN
            * batch size
            ...
        '''
        pass
    
    def save_model(self, path):
        '''
            save models
        '''
        pass
    
    def load_model(self, path):
        '''
            load models
        '''
        pass

# TODO

class ActorNetwork(BaseNetwork):

    '''
        Actor network
    '''

    def __init__(self, **kwargs):
        super(ActorNetwork, self).__init__(kwargs)

    def forward(self, state : GameState) -> (ActionTable, Coordinates):
        '''
            delegate to _inference
        '''
        return self._inference(state)

    def _inference(self, state : GameState):
        '''
            input: state, hidden information
            output: action probabilities
        '''
        common_state, agent_state = state
        pass
    
    def back_prop(self, cost_function):
        pass
    
    def update_params(self, **kwargs):
        '''
            delegate to _update_params
        '''
        self._update_params(**kwargs)
    
    def _update_params(self, **kwargs):
        '''
            input: parameters used to update weights
        '''
        pass

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

# TODO
class CriticNetwork(BaseNetwork):

    '''
        Critic network
    '''

    def __init__(self, **kwargs):
        super(CriticNetwork, self).__init__(kwargs)

    def forward(self, state : GameState, action_probs : ActionTable):
        '''
            delegate to _inference
        '''
        return self._inference(state, action_probs)

    def _inference(self, state : GameState, action_probs : ActionTable):
        '''
            input: state, action probability
            output: Q value
        '''
        common_state, agent_state = state
        pass

    def back_prop(self, cost_function):
        pass

    def update_params(self, **kwargs): 
        '''
            delegate to _update_params
        '''
        self._update_params(**kwargs)
    
    def _update_params(self, **kwargs):
        '''
            input: parameters used to update weights
        '''    
        pass
    
    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass