
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseESN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        density: float = 0.1,
        leaky_alpha: float = 0.1,
        spectral_radius: float = 0.95,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.leaky_alpha = nn.Parameter(torch.tensor(leaky_alpha))

        # Input weights (dense is usually fine for input, but we can make it sparse too if needed)
        # Standard ESNs often have dense input projection. Let's keep it dense for now as user only specified sparse matrix for the network (recurrence usually).
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)

        # Recurrent weights (Sparse)
        # We start with a random sparse matrix
        W_rec = torch.randn(hidden_size, hidden_size)
        mask = torch.rand(hidden_size, hidden_size) < density
        W_rec = W_rec * mask.float()
        
        # Spectral radius scaling
        eigenvalues = torch.linalg.eigvals(W_rec)
        max_eigen = torch.max(torch.abs(eigenvalues))
        if max_eigen > 0:
            W_rec = W_rec * (spectral_radius / max_eigen)
            
        self.W_rec = nn.Parameter(W_rec)
        
        # Readout is NOT a separate layer, output neurons are part of the reservoir.
        # We will assume the first 'output_size' neurons of the reservoir are the output.

    def forward(self, x, h):
        """
        x: [batch, input_size]
        h: [batch, hidden_size]
        """
        # Linear storage
        # W_rec @ h is [hidden, hidden] @ [batch, hidden].T -> [hidden, batch] -> transpose -> [batch, hidden]
        # Common convention: h_next = tanh(W_rec @ h + W_in @ x)
        
        pre_activation = F.linear(h, self.W_rec) + F.linear(x, self.W_in)
        
        # Leaky Integrator ESN:
        # h_new = (1 - alpha) * h + alpha * LeakyReLU(...)
        
        # Ensure alpha is in [0, 1] (though clamping happens in training loop)
        alpha = torch.clamp(self.leaky_alpha, 0.01, 0.99)
        
        update = F.leaky_relu(pre_activation, negative_slope=0.01)
        h_new = (1 - alpha) * h + alpha * update
        
        return h_new

    def get_output(self, h):
        """
        Extracts output from the reservoir state.
        The output neurons are part of the reservoir.
        """
        # Using first output_size neurons
        return h[:, :self.output_size]


    def enforce_spectral_radius(self, target_radius: float = 0.95):
        """
        Rescales the recurrent weights to have the specified spectral radius.
        Returns the computed spectral radius before scaling.
        """
        with torch.no_grad():
            # For large matrices, power iteration is faster, but for 512 eigvals is fine.
            # Convert to CPU for eigvals if on GPU as it can be unstable or slow depending on implementation
            # usually torch.linalg.eigvals is fine on GPU for this size.
            try:
                # abs() of eigenvalues for complex numbers
                eigenvalues = torch.linalg.eigvals(self.W_rec)
                max_eigen = torch.max(torch.abs(eigenvalues))
                
                if max_eigen > 0:
                     scale = target_radius / max_eigen
                     self.W_rec.data.mul_(scale)
                
                return max_eigen.item()
            except Exception as e:
                print(f"Warning: Spectral radius calc failed: {e}")
                return 0.0


