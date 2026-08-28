import numpy as np

def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "mass": 1,  # point mass of bouncing ball (kg)
        "elasticity": 0.8,  # elasticity coefficient 
        "drag_coeff": 0.1,  # drag coefficient
    }
    return params

def dynamics(t, state, params):
    gravity = params["gravity"]
    mass = params["mass"]
    elasticity = params["elasticity"]
    drag_coeff = params["drag_coeff"]
    h = state[0]
    v = state[1]
    # drag modeled as - bv^2

    drag = -(np.sign(v)) * drag_coeff * v ** 2
    state_derivative = np.array([v, -gravity + drag/mass])
    return state_derivative

def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]

    h = state[0]  # indexes entire row "vectorized" if state is (2, N)
    v = state[1]

    kinetic_energy = 0.5 * mass * (v) ** 2
    potential_energy = mass * gravity * h
    return kinetic_energy, potential_energy