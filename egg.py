# python crying
from typing import final

# real stuff
import numpy as np

@final
class reservoir():
    def __init__(self) -> None:
        self.Win = None
        self.Wr = None
        self.x = None
        pass

    def create_reservoir(self, res_size:np.int8, input_size):
        self.Win = np.random.randint(low=0, high=10, size=(res_size, input_size))
        self.Wr = np.random.randint(low=0, high=10, size=(res_size, res_size))

    def step(self):
        self.x = self.Wr @ self.x

        











