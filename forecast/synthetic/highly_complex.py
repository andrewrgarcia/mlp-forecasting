import torch

def generate_highly_nonlinear_series(length):
    data = [0.5] * 10  # Initial values for 10 lags
    for t in range(10, length):
        prev_values = torch.tensor(data[t-10:t])
        # Introduce stronger nonlinearities with constraints
        next_value = (
            torch.sin(prev_values[-1] * prev_values[-2] + prev_values[-3])
            + torch.cos(prev_values[-4] * prev_values[-5])
            + torch.clamp(torch.exp(-torch.abs(prev_values[-6])), max=10)  # Limit exponential growth
            + torch.clamp((prev_values[-7] ** 2) * torch.sign(prev_values[-8]), min=-10, max=10)  # Clamp power term
            + 0.1 * torch.randn(1).item()  # Add noise
        )
        # Ensure the value remains within a reasonable range
        next_value = torch.clamp(next_value, min=-50, max=50)  # Cap final output
        if torch.isnan(next_value) or torch.isinf(next_value):
            print(f"Invalid value at step {t}: {next_value}, replacing with 0.0")
            next_value = 0.0  # Replace with a safe fallback
        data.append(next_value)
    return torch.tensor(data, dtype=torch.float32)