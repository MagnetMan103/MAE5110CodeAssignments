

def integrate(state_traj, timestep, step, t, params, dynamics):
    k1 = dynamics(
            t, state_traj[:, step], params
        )
    k2 = dynamics(
            t + (timestep/2), state_traj[:, step] + k1 * (timestep/2), params
        )
    k3 = dynamics(
            t + (timestep/2), state_traj[:, step] + k2 * (timestep/2), params
        )
    k4 = dynamics(
            t + timestep, state_traj[:, step] + timestep * k3, params
        )
    return state_traj[:, step] + (timestep/6) * (k1 + 2*k2 + 2*k3 + k4)
