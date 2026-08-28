import numpy as np
import matplotlib.pyplot as plt

from models import pendulum as model
#from integrators import explicit_euler as integrator
from integrators import rk4 as integrator
#from models import bouncing as model
# Basic simulation of the pendulum
params = model.generate_params()
params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}


# some set-up
initial_state = np.array([np.pi / 4, 0.0])
#initial_state = np.array([2, 0.0])

timestep = 1e-5
sim_time = 5.0

broken = False


# simulation loop
while not broken:
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    energy_initial = sum(model.calculate_energy(state_traj[:,0], params))
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = integrator.integrate(state_traj, timestep, step, t, params, model.dynamics)
        if state_traj[0][step+1] <= 0 and "elasticity" in params:
            state_traj[1][step+1] = - state_traj[1][step+1] * params["elasticity"]
        if abs(sum(model.calculate_energy(state_traj[:,step + 1], params)) - energy_initial) >= energy_initial/1000:
            print(f"unstable at timestep {timestep} with value {np.abs(state_traj[0][step])}")
            # with euler integration
            # unstable at timestep 2.357947691000002e-05 with value 5.262086604192979

            # with runge-kutta
            # unstable at timestep 0.11388935818035024 with value 5.373195173794081
            broken = True
    timestep *= 1.1
    #broken = True

# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()


# TODO: make a phase portrait plot
plt.figure()
plt.plot(state_traj[0], state_traj[1], label="Phase portrait")
plt.title("Phase portrait")
plt.xlabel("x")
plt.ylabel("x_dot")


plt.show()