"""Minimal mojo-control usage example."""

import numpy as np

import mojo_control as ct


plant = ct.tf([1.0], [1.0, 2.0, 1.0])
closed_loop = ct.feedback(plant, 1)

omega = np.logspace(-2, 2, 500)
response = ct.frequency_response(closed_loop, omega)

time = np.linspace(0.0, 10.0, 2001)
step = ct.step_response(closed_loop, time)

print(response.magnitude.shape)
print(f"{step.outputs[-1]:.6f}")
