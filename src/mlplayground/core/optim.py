import numpy as np

class BatchGD:
    def __init__(self, lr=1e-2):
        self.lr = lr
    
    def step(self, grad_params):
        for W, g in grad_params:
            W -= self.lr * g

class AdamW: 
    def __init__(self, lr=1e-2, alpha=1e-2, beta_1=0.9, beta_2=0.999, delta=1e-8, weight_decay=0):
        self.lr = lr
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.delta = delta
        self.weight_decay = weight_decay

        self.state = {}
        self.steps = 0
        self.W = None

    def get_state(self, param):
        key = id(param)
        if key not in self.state:
            self.state[key] = {
                "m": np.zeros_like(param),
                "v": np.zeros_like(param)
            }
        return self.state[key]

    def step(self, grad_params):

        # update number of steps
        self.steps += 1

        for W, g in grad_params: # i.e. [(W_param, g_param), (b, g_bias)]
            curr_state = self.get_state(W)
            curr_state['m'] = self.beta_1 * curr_state['m'] + (1 - self.beta_1) * g
            curr_state['v'] = self.beta_2 * curr_state['v'] + (1 - self.beta_2) * (g ** 2)
            #print(curr_state['m'])
            #print(curr_state['v'])

            m_hat = curr_state['m'] / (1 - self.beta_1 ** self.steps)
            v_hat = curr_state['v'] / (1 - self.beta_2 ** self.steps)

            W -= self.lr * (self.alpha * (m_hat / (np.sqrt(v_hat) + self.delta)) + self.weight_decay * W)

    def reset_steps(self):
        steps = 0

    def set_schedule_multiplier(self):
        return 0