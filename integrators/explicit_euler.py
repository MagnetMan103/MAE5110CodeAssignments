

def integrate(state_traj, timestep, step, t, params, dynamics):
    return state_traj[:, step] + timestep * dynamics(
        t, state_traj[:, step], params
    )