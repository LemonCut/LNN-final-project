# Lagrangian Neural Networks: Spring Pendulum Dynamics

**PHYS 141 Final Project - UCSD, Spring 2025**  
_Taught by Professor Hongbo Zhao_  
_Project by Christian O'Connor_

## Overview

This project explores **Lagrangian Neural Networks (LNNs)**, a specialized type of neural network that leverages the Euler-Lagrange equations to model physical dynamics with an emphasis on energy conservation. The notebook demonstrates how LNNs can accurately learn and predict the behavior of a spring (elastic) pendulum system from data, and compares their performance against standard multilayer perceptron (MLP) models.

**Inspiration**: This work is heavily inspired by [Sam Greydanus' research on Lagrangian Neural Networks](https://colab.research.google.com/drive/1CSy-xfrnTX28p1difoTA8ulYw0zytJkq).

## Project Structure

The Jupyter notebook (`PHYS_141_Final_Project.ipynb`) contains:

1. **Analytical Modeling** - Deriving the Lagrangian for a spring pendulum system from first principles
2. **Data Generation** - Creating training data using the analytical model
3. **LNN Implementation** - Building and training a Lagrangian Neural Network
4. **Baseline Comparison** - Training a standard MLP for comparison
5. **Results & Visualization** - Comparing model performance through animations and plots

## Physical System: Spring Pendulum

The system models an **elastic/spring pendulum** with:

- **Coordinates**: (θ, x) where θ is the angle from vertical, x is the spring displacement from equilibrium
- **Kinetic Energy**: Combination of rotational and translational components
- **Potential Energy**: Spring potential plus gravitational potential

### Lagrangian Formulation

The analytical Lagrangian L = T - V is expressed as:

```
L(θ, x, θ̇, ẋ) = ½m(ẋ² + (l+x)²θ̇²) - [½kx² - mg(l+x)cos(θ)]
```

Where:

- m = mass
- l = equilibrium spring length
- g = gravitational acceleration
- k = spring constant

## Dependencies

```python
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial
```

**Note**: Gemini Pro 2.5 was used for some code generation assistance.

## Key Features

### Lagrangian Neural Networks (LNN)

- Incorporates physics constraints directly into the network architecture
- Uses Euler-Lagrange equations to ensure energy conservation
- More data-efficient and accurate for physical system modeling
- Better extrapolation capabilities for long-term predictions

### Standard MLP Comparison

- Trained on identical data as the LNN
- Demonstrates the importance of physics-informed architectures
- Shows degradation in accuracy over extended rollouts

## Results

The notebook generates several visualizations including:

1. **Energy Conservation Plots** - Comparing how well each model preserves total energy
2. **Trajectory Comparisons** - Side-by-side animations of analytical, LNN, and MLP predictions
3. **State Variable Evolution** - Time series plots of angular position, spring displacement, and their derivatives
4. **Rollout Analysis** - Long-term prediction accuracy assessment

### Key Finding

The LNN significantly outperforms the standard MLP in:

- **Accuracy**: Closer adherence to the analytical solution
- **Energy Conservation**: Better preservation of system energy over time
- **Long-term Stability**: More reliable predictions during extended rollouts

The standard MLP particularly struggles with angular position and velocity predictions, while the LNN maintains accuracy across all state variables.

## Usage

1. Open `PHYS_141_Final_Project.ipynb` in Jupyter Notebook or JupyterLab
2. Run cells sequentially to:
   - Derive the analytical model
   - Generate training data
   - Train both LNN and MLP models
   - Visualize and compare results
3. Experiment with different hyperparameters, initial conditions, or system parameters

## Educational Value

This project demonstrates:

- **Physics-Informed Machine Learning**: How incorporating domain knowledge improves model performance
- **Classical Mechanics**: Practical application of Lagrangian mechanics
- **Neural Network Design**: Comparison of standard vs. physics-constrained architectures
- **Scientific Computing**: Using JAX for automatic differentiation and numerical integration

## References

- Greydanus, S., Dzamba, M., & Yosinski, J. (2019). Hamiltonian Neural Networks. _Advances in Neural Information Processing Systems_, 32.
- [Sam Greydanus' LNN Colab Notebook](https://colab.research.google.com/drive/1CSy-xfrnTX28p1difoTA8ulYw0zytJkq)

## License

This project is for educational purposes as part of PHYS 141 coursework at UCSD.

---

_For questions or collaboration, please contact the project author._
