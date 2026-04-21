import torch
import torch.nn as nn

# TODO
# - batching support
# - 
# - 
# - 
# - 
# - 
# - 

class LayerHebb(nn.Module):
    def __init__(self, in_dims:int, out_dims:int, batch_size:int, n_coeff:int = 5):
        super().__init__()
        self.in_dims : int = in_dims
        self.out_dims : int = out_dims
        self.batch_size : int = batch_size
        self.n_coeff : int = n_coeff
        self.register_buffer("weights", torch.empty(batch_size, in_dims, out_dims))
        self.weights : torch.Tensor
        self.reset_weights()

        self.register_buffer("x", torch.zeros(in_dims))
        self.x: torch.Tensor
        self.register_buffer("y", torch.zeros(out_dims))
        self.y: torch.Tensor

        # Initialize coefficients
        self.coefficients : nn.Parameter = nn.Parameter(torch.empty(in_dims, out_dims, n_coeff))
        _ = nn.init.uniform_(self.coefficients, -1.0, 1.0)


    def reset_weights(self):
        _ = nn.init.uniform_(self.weights, -0.1, 0.1)

    def forward(self, x:torch.Tensor):
        # save x to update the weights | (B, 1, in_dims)
        self.x = x.clone()

        # (B, 1, out_dims)
        self.y = torch.bmm(x.unsqueeze(1), self.weights).squeeze(1)
        return self.y

    def update_weights(self):
        """
        Delta_W = lr * ( (in, out)*((in,1)@(1,out)) + (in, out)*(in, 1) +   (in, out)*(1, out) +  ()*())
                             A                         row-wise broadcast    col-wise broadcast 
        """
        x_col = self.x.view(self.batch_size, self.in_dims, 1)  # (B, in_dims, 1)
        y_row = self.y.view(self.batch_size, 1, self.out_dims)  # (B, 1, out_dims)
        
        with torch.no_grad():
            # Hebbian term: x * y | (B, in_dims, out_dims)
            term1 = self.coefficients[..., 0] * torch.bmm(x_col, y_row)
            # Input term: x
            term2 = self.coefficients[..., 1] * x_col
            # Output term: y
            term3 = self.coefficients[..., 2] * y_row
            # Bias/Fixed term
            term4 = self.coefficients[..., 3]
            # Learning Rate
            eta   = self.coefficients[..., 4]
            
            # In-place update to the weight parameter 
            _ = self.weights.add_(eta * (term1 + term2 + term3 + term4))


class HebbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers : nn.ModuleDict = nn.ModuleDict({
            'layer1': LayerHebb(10, 20, 5),
            'layer2': LayerHebb(20, 10, 5)
            })
        self.act : nn.Tanh = nn.Tanh()
    
    def forward(self, x):
        # x dimensions should be (B, 1, in_dims)
        x = self.layers['layer1'](x)
        x = self.act(x)
        x = self.layers['layer2'](x)
        return x

    def update_layers(self):
        for layer in self.layers.values():
            layer.update_weights()


if __name__ == "__main__":
     
    model = HebbModel()
    with torch.no_grad():
        print(model.forward(torch.rand(1, 10)))
        model.update_layers()

















