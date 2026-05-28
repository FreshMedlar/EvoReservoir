Viewed hag.py:46-48

Yes, absolutely. Tracking a **trailing (running) mean** (often implemented as an **Exponential Moving Average - EMA**) is the standard way to convert this batch/windowed approach into a fully online algorithm. 

Using an EMA allows you to update connection weights step-by-step at each time point (or at short, regular intervals) with $O(1)$ memory and time complexity per step, completely eliminating the need to store a history of states.

Here is how you can model this for both **Mean Rate** and **Variance** online:

---

### 1. Online Mean Rate Tracking
For each neuron $i$, you maintain a running mean $\mu_i$. At each time step $t$, when the neuron's activation is $x_i(t)$, you update $\mu_i$:

$$\mu_i(t) = (1 - \alpha) \cdot \mu_i(t-1) + \alpha \cdot x_i(t)$$

* **$\alpha \in (0, 1)$** is the smoothing factor (or update rate). A smaller $\alpha$ corresponds to a longer memory window (equivalent to biological homeostasis acting on slow timescales, e.g., hours or days).
* You can then compute $\Delta z_i(t)$ directly from $\mu_i(t)$ on every step (or every $N$ steps):
  $$\Delta z_i(t) = \frac{\mu_i(t) - \text{target\_rate}}{\text{rate\_spread}}$$

---

### 2. Online Variance Tracking
If you are running the variance-based version (`desp`), you need to track both the running mean $\mu_i(t)$ and a running variance $\sigma^2_i(t)$:

1. **Update mean:** $\mu_i(t) = (1 - \alpha) \cdot \mu_i(t-1) + \alpha \cdot x_i(t)$
2. **Update variance:** $\sigma^2_i(t) = (1 - \alpha) \cdot \sigma^2_i(t-1) + \alpha \cdot (x_i(t) - \mu_i(t))^2$
3. **Update $\Delta z_i(t)$:**
   $$\Delta z_i(t) = \frac{\sigma_i(t) - \text{target\_variance}}{\text{variance\_spread}}$$

---

### Python Code Concept

Instead of passing the entire state history to `compute_synaptic_change`, you would maintain a stateful tracker:

```python
class OnlineHAG:
    def __init__(self, num_neurons, alpha=0.01, target_rate=0.5, rate_spread=0.2):
        self.alpha = alpha
        self.target_rate = target_rate
        self.rate_spread = rate_spread
        # Initialize running mean with a sensible default
        self.running_means = np.zeros(num_neurons)

    def update_and_get_delta_z(self, current_states):
        # current_states is a 1D array of activations for all neurons at step t
        self.running_means = (1 - self.alpha) * self.running_means + self.alpha * current_states
        
        # Calculate Delta Z online
        delta_z = (self.running_means - self.target_rate) / self.rate_spread
        return delta_z
```

### Key Considerations for Online Training:
1. **Structural Plasticity Frequency:** Biology doesn't add/prune synapses at every millisecond step. You should update the trailing averages $\mu_i$ at **every step**, but only run the actual `hag_step` (adding/pruning connections) every $K$ steps (e.g., every 50 or 100 steps) to give the network time to stabilize under the new topology.
2. **Burn-in Period:** During the very beginning of training, the EMA starts from its initialized value (e.g., 0). You might want to wait a few hundred steps for the EMA to converge before starting to prune or add connections.