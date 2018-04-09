from utils import GameState, AgentState, CommonState, ActionTable

# TODO
'''
    Actor network
'''
class ActorNetwork():

    def __init__(self, **kwargs):
        '''
        Init actor network
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

    '''
        delegate to _inference
    '''
    def forward(self, state : GameState):
        return self._inference(state)

    '''
        input: state, hidden information
        output: action probabilities
    '''
    def _inference(self, state : GameState):
        common_state, agent_state = state
    
    '''
        delegate to _update_params
    '''
    def update_params(self, **kwargs):
        self._update_params(**kwargs)
    
    '''
        input: parameters used to update weights
    '''
    def _update_params(self, **kwargs):
        pass

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

# TODO
'''
    Critic network
'''
class CriticNetwork():

    def __init__(self, **kwargs):
        '''
        Init Critic network
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

    '''
        delegate to _inference
    '''
    def forward(self, state : GameState, action_probs : ActionTable):
        return self._inference(state, actions_prob)

    '''
        input: state, action probability
        output: Q value
    '''
    def _inference(self, state : GameState, action_probs : ActionTable):
        common_state, agent_state = state
    
    '''
        delegate to _update_params
    '''
    def update_params(self, **kwargs):
        self._update_params(**kwargs)
    
    '''
        input: parameters used to update weights
    '''
    def _update_params(self, **kwargs):
        pass
    
    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass