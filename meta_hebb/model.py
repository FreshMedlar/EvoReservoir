import torch

# TODO
# - self.weights random initialization
# - variable coefficients number

class LayerHebb():
    def __init__(self, in_dims, out_dims):
        self.coeff = 5
        self.weights = torch.rand(in_dims, out_dims)
        self.coefficients = torch.randn(out_dims, in_dims, self.coeff)



class Hebb():
    def __init__(self):
        return

    def forward(self):

        pass


if __name__ == "__main__":
    
    model = Hebb()

    
