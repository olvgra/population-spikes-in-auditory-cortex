import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.classes.A1Model import A1Model
from src.classes.Params import Params
from src.classes.Plotter import Plotter


def main():
    # --- Get Model Parameters --- #
    params = Params(sim_duration=0.2)

    # --- Initialise the model --- #
    a1_model = A1Model(params)

    act = np.zeros((a1_model.params.P, int(a1_model.params.sim_duration/a1_model.params.dt)))

    # --- Define the Stimuli --- #
    for i in range(15):
        col = i + 1
        a1_model.add_stimulus(col, 6, 0.05, 0.1)
        a1_model.add_stimulus(8, 2, 0.095, 0.145)

        a1_model.run()

        act[i] = np.mean(a1_model.activity.E[:,7,:], axis=1)

    plotter = Plotter(a1_model)
    plotter.figure_10(act)


if __name__ == '__main__':
    main()
