import numpy as np

from src.classes.A1Model import A1Model
from src.classes.Params import Params
from src.classes.Plotter import Plotter
from src.utils.Helper import set_locking_stimuli_for_all_columns

def main():
    params = Params(sim_duration=4.5+1.3)
    params.time_constant = params.tau_rec

    a1_model = A1Model(params)

    # --- All neuron locking suppression --- #
    # --- Run for all neurons --- #
    set_locking_stimuli_for_all_columns(a1_model, 1.56, 1.3, 0.01)
    a1_model.add_stimulus(8, 0.73, 1 + 1.3, 4.5 + 1.3)
    a1_model.run(all_neurons=True)

    index = int((1.3*a1_model.params.tau_rec)/a1_model.params.dt)
    time_steps = a1_model.activity.E[index:,:,:].shape[0]

    activity = np.mean(a1_model.get_activity().E[index:,7,:], axis = 1)

    plotter = Plotter(a1_model)
    plotter.figure_11(activity, time_steps)

    # --- Spont neuron locking suppression --- #
    # --- Run for spont neurons --- #
    a1_model.update_params(sim_duration=4.5+1.3)
    set_locking_stimuli_for_all_columns(a1_model, 2.78, 1.3, 0.01)
    a1_model.add_stimulus(8, 1.37, 2.3, 4.5+1.3)
    a1_model.run()

    index = int((1.3*a1_model.params.tau_rec)/a1_model.params.dt)
    time_steps = a1_model.activity.E[index:,:,:].shape[0]

    activity = np.mean(a1_model.get_activity().E[index:,7,:], axis = 1)
    plotter.figure_11(activity, time_steps)

    # Second NC example
    # --- Run for spont neurons --- #
    set_locking_stimuli_for_all_columns(a1_model, 2.78, 1.3, 0.01)
    a1_model.add_stimulus(8, 1.37, 1.4+1.3, 4.5+1.3)
    a1_model.run()

    index = int((1.3*a1_model.params.tau_rec)/a1_model.params.dt)
    time_steps = a1_model.activity.E[index:,:,:].shape[0]

    activity = np.mean(a1_model.get_activity().E[index:,7,:], axis = 1)
    plotter.figure_11(activity, time_steps)


if __name__ == '__main__':
    main()