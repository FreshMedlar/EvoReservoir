import torch

from meta_hebb.model import HebbModel

def get_batch(batch_size, seq_len, x_dim):
    # 1. Sample random weights w for each batch element (B, x_dim, 1)
    # The model must figure out these weights purely from context!
    w = torch.randn(batch_size, x_dim, 1, device=device)
    
    # 2. Sample random inputs x (B, seq_len, x_dim)
    x = torch.randn(batch_size, seq_len, x_dim, device=device)
    
    # 3. Compute y = x @ w (B, seq_len, 1)
    # We add a tiny bit of noise usually, but for pure ICL demonstration, clean is fine.
    y = x @ w 
    
    return x, y


def main():

    model = HebbModel()











if __name__ == "__main__":
    main()

