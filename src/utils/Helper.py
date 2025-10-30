import numpy as np


def shift_array(arr, shift):
    rolled = np.roll(arr, shift, axis=0)
    if shift > 0:
        rolled[:shift] = 0
    elif shift < 0:
        rolled[shift:] = 0
    return rolled


def relu(x):
    return np.maximum(x, 0)


def set_complex_sound_stimuli_C(model):
    model.add_stimulus(2, 5, 1.2, 1.3)
    model.add_stimulus(3, 6, 0.5, 0.8)
    model.add_stimulus(4, 7, 1.3, 1.57)
    model.add_stimulus(4, 6, 1.89, 2.08)
    model.add_stimulus(6, 5, 1.5, 1.76)
    model.add_stimulus(8, 4, 0.28, 0.48)
    model.add_stimulus(8, 3, 1.65, 1.78)
    model.add_stimulus(8, 8, 2.12, 2.38)
    model.add_stimulus(10, 5, 0.85, 0.98)
    model.add_stimulus(12, 4, 0.12, 0.26)
    model.add_stimulus(14, 7, 1.08, 1.32)
    model.add_stimulus(14, 7, 2.38, 2.5)

def set_complex_sound_stimuli_B(model):
    model.add_stimulus(2, 5, 1.2, 1.3)
    model.add_stimulus(3, 4, 0.15, 0.25)
    model.add_stimulus(4, 5, 0.85, 0.95)
    model.add_stimulus(4, 6, 1.89, 2.08)
    model.add_stimulus(6, 7, 2.35, 2.5)
    model.add_stimulus(8, 6, 0.45, 0.75)
    model.add_stimulus(8, 3, 1.65, 1.78)
    model.add_stimulus(10, 8, 2.1, 2.35)
    model.add_stimulus(11, 7, 1.3, 1.55)
    model.add_stimulus(12, 7, 1.05, 1.3)
    model.add_stimulus(14, 4, 0.25, 0.45)
    model.add_stimulus(14, 5, 1.45, 1.75)

def set_locking_stimuli_for_all_columns(model, A, isi, stimulus_duration):
    num_stimuli = int((model.params.sim_duration/model.params.time_constant) / isi + 1)
    first_stimulus_time = ((model.params.sim_duration/model.params.time_constant) - (num_stimuli - 1) * isi) / 2
    stimulus_times = [first_stimulus_time + i * isi for i in range(num_stimuli)]
    for column in range(model.params.P):
        for stim_time in stimulus_times:
                model.add_stimulus(column, A, stim_time, stim_time + stimulus_duration)
    return stimulus_times, first_stimulus_time

def set_stimuli_for_all_columns(model, A, start, stimulus_duration):
    for column in range(model.params.P):
          model.add_stimulus(column, A, start, start + stimulus_duration)
