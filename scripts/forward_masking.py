import numpy as np

from src.classes.A1Model import A1Model
from src.classes.Params import Params
from src.classes.Plotter import Plotter


def main():
    # --- Get Model Parameters --- #
    params = Params(sim_duration=4.5, seed=47)
    params.time_constant = params.tau_rec

    # --- Initialise Model --- #
    a1_model = A1Model(params)

    # --- Define Intervals --- #
    ISI = [0.125, 0.25, 0.5, 1, 2, 4]
    A_vals = [10, 5.5, 4]
    c_vals = [7, 5, 4, 3, 2]

    # --- Initialise Outputs --- #
    outputs = []
    time_axes = []
    ratio_A = np.zeros((3, 6))
    ratio_F = np.zeros((5, 6))

    for i, A in enumerate(A_vals):
        for j, isi in enumerate(ISI):
            # --- Define the Stimuli --- #
            a1_model.add_stimulus(8, A, 0, 0.05)
            a1_model.add_stimulus(8, A, isi, isi + 0.05)

            # --- Run Simulation --- #
            a1_model.run()

            activity = a1_model.get_activity()

            # --- Figure 6A: Network Activity Over ISIs --- #
            if A == 10:
                output_activity = activity.E
                outputs.append(output_activity)

                time_steps = output_activity.shape[0]
                time_axis = np.arange(time_steps) * (a1_model.params.dt / a1_model.params.tau_rec)
                time_axes.append(time_axis)

            E_activity_masker = activity.E[:int(0.05 * a1_model.params.tau_rec / a1_model.params.dt), :, :]
            E_activity_probe = activity.E[int(isi * a1_model.params.tau_rec / a1_model.params.dt):int((isi + 0.05) * a1_model.params.tau_rec / a1_model.params.dt), :, :]

            # --- Figure 6B: P2:P1 Over ISIs & Amplitude --- #
            P1 = np.mean(np.mean(E_activity_masker[:, 7, :], axis=1), axis=0)
            P2 = np.mean(np.mean(E_activity_probe[:, 7, :], axis=1), axis=0)
            ratio_A[i, j] = P2 / P1

            # --- Figure 6C: P2:P1 Over ISIs & Frequency --- #
            if A == 10:
                for k, F in enumerate(c_vals):
                    P1 = np.mean(np.mean(E_activity_masker[:, F, :], axis=1), axis=0)
                    P2 = np.mean(np.mean(E_activity_probe[:, F, :], axis=1), axis=0)
                    ratio_F[k, j] = P2 / P1

    # --- Plot Figure 6 --- #
    plotter = Plotter(a1_model)
    plotter.figure_6(outputs, time_axes, ratio_A, ratio_F, ISI, A_vals, c_vals)


if __name__ == '__main__':
    main()