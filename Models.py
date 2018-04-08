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
    def __init__(self):
        pass

    '''
        delegate to _inference
    '''
    def forward(self, s, h0):
        return self._inference(s, h0)

    '''
        input: state, hidden information
        output: action probabilities
    '''
    def _inference(self, s, h0):
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
    def __init__(self):
        pass

    '''
        delegate to _inference
    '''
    def forward(self, s, h0):
        return self._inference(s, h0)

    '''
        input: state, action probability
        output: Q value
    '''
    def _inference(self, s, a):
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