# TODO
'''
    Actor network
'''
class ActorNetwork():

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
    def __init__(self, **kwargs):
        pass

    '''
        delegate to _inference
    '''
    def forward(self, state, agents):
        return self._inference(state, agents)

    '''
        input: state, hidden information
        output: action probabilities
    '''
    def _inference(self, state, agents):
        pass
    
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
    def __init__(self, **kwargs):
        pass

    '''
        delegate to _inference
    '''
    def forward(self, state, agents, actions):
        return self._inference(state, agents, actions)

    '''
        input: state, action probability
        output: Q value
    '''
    def _inference(self, state, agents, actions):
        pass
    
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