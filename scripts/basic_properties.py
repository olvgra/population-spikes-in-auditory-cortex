import numpy as np

from src.classes.A1Model import A1Model
from src.classes.Params import Params
from src.classes.Plotter import Plotter


def main():
    # --- Get Model Parameters --- #
    params = Params(sim_duration=0.5)

    # --- Initialise the model --- #
    a1_model = A1Model(params)

    # --- Define the Stimuli --- #
    a1_model.add_stimulus(8, 4, 0.3, 0.5)

    # --- Run Simulation for Figure 2 --- #
    a1_model.run()

    # --- Plot Figure 2 --- #
    plotter = Plotter(a1_model)
    plotter.figure_2()


    # --- Run Simulation for Figure 3 --- #
    E_act_store = []
    s_act_store = []

    for i, A in enumerate(np.linspace(10, 1, 7)):
        # --- Update Model Parameters --- #
        a1_model.update_params(sim_duration=0.07, dt=0.0001)

        # --- Update the Stimuli --- #
        a1_model.add_stimulus(8, A, 0.01, 0.07)

        # --- Run Simulation --- #
        a1_model.run()

        # --- Store Activity --- #
        sensory_input = a1_model.get_sensory_input_matrix()
        E_act_store.append(np.mean(a1_model.activity.E, axis=2))
        if i % 2 == 1:
            s_act_store.append(np.max(np.mean(sensory_input[:, :, a1_model.spont_index:a1_model.params.N_E], axis=2), axis=0))

    # --- Plot Figure 3 --- #
    plotter.figure_3(np.array(E_act_store), np.array(s_act_store))


    # --- Run Simulation for Figure 4 --- #

    # --- Update the Stimuli --- #
    weak_stimulus = 2
    a1_model.add_stimulus(8, weak_stimulus, 0.01, 0.07)

    # --- Run Simulation --- #
    a1_model.run()

    # --- Store the Activity --- #
    activity = a1_model.get_activity()
    E_weak = activity.E[:, 7, :]
    I_weak = activity.I[:, 7, :]

    # --- Update the Stimuli --- #
    strong_stimulus = 6
    a1_model.add_stimulus(8, strong_stimulus, 0.01, 0.07)

    # --- Run Simulation --- #
    a1_model.run()

    # --- Store the Activity --- #
    activity = a1_model.get_activity()
    E_strong = activity.E[:, 7, :]
    I_strong = activity.I[:, 7, :]

    A_range = np.arange(0, 10, 0.2)
    output = np.full((len(A_range), 3), np.nan)

    for i in range(len(A_range)):
        # --- Update the Stimuli --- #
        a1_model.add_stimulus(8, A_range[i], 0.01, 0.07)

        # --- Run Simulation --- #
        a1_model.run()

        # --- Store the Activity --- #
        activity = a1_model.get_activity()
        E = activity.E[:, 7, :]
        I = activity.I[:, 7, :]

        E_mean = np.mean(E, axis=1)
        I_mean = np.mean(I, axis=1)

        if np.max(E_mean) > 20:
            E_diff = np.diff(np.mean(E, axis=1), axis=0)
            I_diff = np.diff(np.mean(I, axis=1), axis=0)

            E_thrshld = 0.4 * np.max(E_diff)
            I_thrshld = 0.4 * np.max(I_diff)

            E_onset = (np.where(E_diff >= E_thrshld)[0][0]) * a1_model.params.dt
            I_onset = (np.where(I_diff >= I_thrshld)[0][0]) * a1_model.params.dt

            # Take into account stimulus start time
            E_latency = E_onset - 0.01
            I_latency = I_onset - 0.01

            # Convert to seconds
            E_latency = E_latency * 1000
            I_latency = I_latency * 1000

            output[i, :] = A_range[i], E_latency, I_latency

    # --- Plot Figure 4 --- #
    plotter.figure_4(E_weak, I_weak, E_strong, I_strong, output)

if __name__ == '__main__':
    main()