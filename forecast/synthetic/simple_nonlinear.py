import torch

def generate_nonlinear_series(length):
    data = [0.5] * 10  # Initial values for 10 lags
    for t in range(10, length):
        prev_values = torch.tensor(data[t-10:t])
        next_value = torch.sin(prev_values[-1] * prev_values[-2]) + torch.exp(-prev_values[-3]**2) + 0.1 * torch.randn(1).item()
        data.append(next_value)
    return torch.tensor(data, dtype=torch.float32)