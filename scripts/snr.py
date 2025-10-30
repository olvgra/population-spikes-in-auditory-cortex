import numpy as np

from src.classes.A1Model import A1Model
from src.classes.Params import Params
from src.classes.Plotter import Plotter
from src.utils.Helper import set_stimuli_for_all_columns


def main():
    ratios = [30, 24, 18, 12, 6]
    levels = 5
    thresholds = np.array([1.2, 1.21, 1.22, 1.25, 1.34])*1.2

    # --- Get Model Parameters --- #
    params = Params(sim_duration=1)

    # --- Initialise the model --- #
    a1_model = A1Model(params)

    act = np.zeros((len(ratios), levels))

    # Run simulation for each ratio and level
    for i, ratio in enumerate(ratios):
        min_signal = thresholds[i]
        max_noise = 1.08
        noise_range = np.linspace(min_signal**2/ratio, max_noise, levels)
        for j, noise in enumerate(noise_range):
            set_stimuli_for_all_columns(a1_model, noise, 0, 1)
            a1_model.add_stimulus(8, noise*ratio, 0.9, 0.95)
            a1_model.run()
            index = 0
            activity = a1_model.get_activity().E[index:, 7, :]
            E = np.mean(activity, axis = 1)
            max_E = np.max(E, axis = 0)
            act[i, j] = max_E

    plotter = Plotter(a1_model)
    plotter.snr(act)

    # --- FREQUENCY RESPONSE AREA --- #
    levels = 10
    act = np.zeros((5, levels, 15))
    ratios = [30, 24, 18, 12, 6]
    for j, ratio in enumerate(ratios):
        max_noise = 1.08
        noise_range = np.linspace(0, max_noise, levels)
        for i, noise in enumerate(noise_range):
            set_stimuli_for_all_columns(a1_model, noise, 0, 1)
            a1_model.add_stimulus(8, noise * ratio, 0.9, 0.95)
            a1_model.run()
            act[j, i] = np.max(np.mean(a1_model.get_activity().E, axis = 2), axis = 0)

    plotter.fra(act, ratios)


if __name__ == "__main__":
    main()