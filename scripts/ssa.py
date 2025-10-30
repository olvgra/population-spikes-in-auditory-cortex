import numpy as np
import math

from src.classes.A1Model import A1Model
from src.classes.Params import Params
from src.classes.Plotter import Plotter


def main():
    # --- Get Model Parameters --- #
    params = Params(sim_duration=35, seed=47)

    # --- Initialise the model --- #
    a1_model = A1Model(params)

    ISI = 0.35

    time_steps = np.arange(0, params.sim_duration, ISI)  # Stimulus times
    stimuli_type = np.zeros((2, int(params.sim_duration / ISI))) # 1 - Standard, 0 - Deviant
    mean_x = np.zeros((int(params.sim_duration / params.dt), params.P))
    mean_E = np.zeros(int(params.sim_duration / params.dt))

    frequencies = [[8, 10], # f1
                   [10, 8]] # f2

    for j, f in enumerate(frequencies):
        for i, t in enumerate(time_steps):
            if np.random.rand() < 0.9:  # 90% of the time, present standard stimulus
                a1_model.add_stimulus(f[0], 5, start=t, stop=t+0.05)
                stimuli_type[j, i] = 1
            else:  # 10% of the time, present deviant stimulus
                a1_model.add_stimulus(f[1], 5, start=t, stop=t+0.05)
                stimuli_type[j, i] = 0

        # --- Run Simulation --- #
        a1_model.run()

        if(j == 0):
            x_resources = a1_model.get_activity().x
            mean_x = np.mean(x_resources[:, :, :], axis=2).T

            mean_E = np.mean(a1_model.get_activity().E[:, 8, :], axis=1)

        E_activity = a1_model.get_activity().E

        if(j == 0):
            deviant_responses1 = np.zeros((100, np.where(stimuli_type[j] == 0)[0].size))
            standard_responses1 = np.zeros((100, np.where(stimuli_type[j] == 1)[0].size))

        if(j == 1):
            deviant_responses2 = np.zeros((100, np.where(stimuli_type[j] == 0)[0].size))
            standard_responses2 = np.zeros((100, np.where(stimuli_type[j] == 1)[0].size))

        di = 0
        si = 0

        for i, t in enumerate(time_steps):
            if(stimuli_type[j, i] == 0):
                if(j == 0):
                    deviant_responses1[:, di] = np.mean(E_activity[math.ceil((t) / params.dt) : math.ceil((t+0.1) / params.dt), 8, :], axis=1)
                else:
                    deviant_responses2[:, di] = np.mean(E_activity[math.ceil((t) / params.dt) : math.ceil((t+0.1) / params.dt), 8, :], axis=1)

                di = di+1

            if(stimuli_type[j, i] == 1):
                if(j == 0):
                    standard_responses1[:, si] = np.mean(E_activity[math.ceil((t) / params.dt) : math.ceil((t+0.1) / params.dt), 8, :], axis=1)
                else:
                    standard_responses2[:, si] = np.mean(E_activity[math.ceil((t) / params.dt) : math.ceil((t+0.1) / params.dt), 8, :], axis=1)
                si = si+1

    figure_data = {
        "time_steps": time_steps,
        "stimuli_type": stimuli_type,
        "mean_E": mean_E,
        "mean_x": mean_x,
        "deviant_responses1": deviant_responses1,
        "standard_responses1": standard_responses1,
        "deviant_responses2": deviant_responses2,
        "standard_responses2": standard_responses2
    }

    plotter = Plotter(a1_model)
    plotter.ssa(figure_data)


if __name__ == "__main__":
    main()