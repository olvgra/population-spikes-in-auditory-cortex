from src.classes.A1Model import A1Model
from src.classes.Params import Params
from src.classes.Plotter import Plotter
from src.utils.Helper import set_complex_sound_stimuli_C, set_complex_sound_stimuli_B

def main():
    # --- Get Model Parameters --- #
    params = Params(sim_duration=2.5)
    params.time_constant = params.tau_rec

    # --- Initialise the model --- #
    a1_model = A1Model(params)

    # --- Define the Stimuli --- #
    set_complex_sound_stimuli_B(a1_model)

    # --- Run Simulation --- #
    a1_model.run()

    # --- Plot Results --- #
    plotter = Plotter(a1_model)
    plotter.figure_13()

    # --- Define the Stimuli --- #
    set_complex_sound_stimuli_C(a1_model)

    # --- Run Simulation --- #
    a1_model.run()

    # --- Plot Results --- #
    plotter = Plotter(a1_model)
    plotter.figure_13()


if __name__ == '__main__':
    main()