import numpy as np

class RandomAgent:
    def __init__(
        self,
        env_params):

        self.env_params=env_params

    def get_action(self, candidates) -> int:
        args=np.arange(self.env_params['N'])
        np.random.shuffle(args)
        return args[:self.env_params['K']]

    def eval_on(self) :
        pass
    
    def eval_off(self) :
        pass

    def observe(self, candidates, action, choice, reward, done) -> None:
        pass

    def observe_candidates(self, user, candidates: np.ndarray) -> None:
        pass
