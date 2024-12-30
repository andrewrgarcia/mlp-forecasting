### README: `synthetic` Module

The `synthetic` module provides utility functions for generating synthetic time series data with varying degrees of nonlinearity. These functions are designed to facilitate experimentation, testing, and benchmarking of machine learning and deep learning models for time series forecasting tasks.

---

### Functions

#### `generate_nonlinear_series(length: int) -> torch.Tensor`
Generates a basic nonlinear time series.

- **Description**:
  - Uses sinusoidal and exponential transformations with minimal noise.
  - Suitable for initial testing of models on simple nonlinear data.

- **Parameters**:
  - `length` (int): The total length of the time series to generate.

- **Returns**:
  - A `torch.Tensor` containing the generated time series.

- **Example**:
  ```python
  series = generate_nonlinear_series(1000)
  ```

---

#### `generate_moderately_nonlinear_series(length: int) -> torch.Tensor`
Generates a moderately nonlinear time series.

- **Description**:
  - Uses sinusoidal, exponential, and cosine transformations applied to the lagged values.
  - Introduces mild randomness to mimic real-world noise.
  - Suitable for testing models on data with moderate nonlinearity.

- **Parameters**:
  - `length` (int): The total length of the time series to generate.

- **Returns**:
  - A `torch.Tensor` containing the generated time series.

- **Example**:
  ```python
  series = generate_moderately_nonlinear_series(1000)
  ```

---

#### `generate_highly_nonlinear_series(length: int) -> torch.Tensor`
Generates a highly nonlinear time series with stronger and more complex interactions.

- **Description**:
  - Incorporates stronger nonlinear dependencies by combining sinusoidal, cosine, and exponential functions with power terms.
  - Clamps extreme values to ensure numerical stability.
  - Suitable for testing models on challenging data with high nonlinearity and bounded constraints.

- **Parameters**:
  - `length` (int): The total length of the time series to generate.

- **Returns**:
  - A `torch.Tensor` containing the generated time series.

- **Example**:
  ```python
  series = generate_highly_nonlinear_series(1000)
  ```

---

### Key Features

- **Controlled Complexity**:
  - The module provides functions with varying levels of nonlinearity, enabling progressive testing from simple to complex scenarios.

- **Noise Injection**:
  - All functions include random noise to simulate real-world unpredictability.

- **Numerical Stability**:
  - The `generate_highly_nonlinear_series` function ensures stability by clamping extreme values and handling invalid computations.

---

### Usage

```python
import synthetic

# Generate a moderately nonlinear time series
moderate_series = synthetic.generate_moderately_nonlinear_series(500)

# Generate a highly nonlinear time series
highly_nonlinear_series = synthetic.generate_highly_nonlinear_series(500)

# Generate a simple nonlinear time series
basic_nonlinear_series = synthetic.generate_nonlinear_series(500)
```

---

### Notes

- The initial values for the first 10 lags are hardcoded to `0.5` in all functions.
- Ensure the `length` parameter is greater than 10 to accommodate lagged dependencies.

This module is intended for synthetic data generation only. Real-world datasets should be used for validating model performance in production scenarios.