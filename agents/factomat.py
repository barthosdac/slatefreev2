import numpy as np

class FactoMatAgent:
    def __init__(
        self,
        env_params):
        self.user=None
        self.env_params=env_params

    def get_action(self, candidates) -> int:
        scores=(self.user[None,:]*candidates).sum(axis=-1)
        return scores.argsort()[-1:-self.env_params['K']-1:-1]

    def eval_on(self) :
        pass
    
    def eval_off(self) :
        pass

    def observe(self, candidates, action, choice, reward, done) -> None:
        pass

    def observe_candidates(self, user, candidates: np.ndarray) -> None:
        self.user=user
