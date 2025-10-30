import numpy as np
from scipy.integrate import solve_ivp

from src.classes.Stimuli import Stimuli
from src.utils.Helper import shift_array, relu

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class State:
    """Tracks model state, including initialisation and equilibration status."""
    def __init__(self):
        self.initialising = True
        self.equilibrated = False

class Equilibrium:
    """Stores equilibrium states for neural populations and synaptic resources."""
    def __init__(self):
        self.E = self.I = self.x = self.y = None
        self.spont_index = None

class Activity:
    """Tracks model activity over time."""
    def __init__(self):
        self.E = self.I = self.x = self.y = None

    def copy(self):
        """Returns a copy of the activity."""
        new_activity = Activity()
        new_activity.E = self.E.copy() if self.E is not None else None
        new_activity.I = self.I.copy() if self.I is not None else None
        new_activity.x = self.x.copy() if self.x is not None else None
        new_activity.y = self.y.copy() if self.y is not None else None
        return new_activity


class A1Model:
    def __init__(self, params):
        # Store parameters
        self.params = params
        self.stimuli = Stimuli(params.P)
        self.state = State()
        self.equilibrium = Equilibrium()
        self.activity = Activity()
        self.spont_index = 0

        self._initialise_model()

    def _initialise_model(self):
        """Sets up populations, resources, external inputs, and synaptic weights."""
        self.E = np.zeros((self.params.P, self.params.N_E))
        self.I = np.zeros((self.params.P, self.params.N_I))
        self.x = np.full((self.params.P, self.params.N_E), self.params.initial_x)
        self.y = np.full((self.params.P, self.params.N_I), self.params.initial_y)

        np.random.seed(self.params.seed)
        self.e_E = np.sort(np.random.uniform(self.params.e_E_1, self.params.e_E_NE, self.params.N_E)) + self.params.delta_e
        self.e_I = np.sort(np.random.uniform(self.params.e_I_1, self.params.e_I_NI, self.params.N_I)) + self.params.delta_e
        
        if hasattr(self.params, 'e_E'):
            self.e_E = self.params.e_E

        self._initialise_synaptic_weights()
        self._initialise_sim_duration()

        self._equilibrate()
        self.state.initialising = False

    def _initialise_synaptic_weights(self):
        """Computes normalised synaptic weights."""
        p = self.params
        self.J_0_EE, self.J_0_EI = round(p.J_0_EE / p.N_E, 10), round(p.J_0_EI / p.N_I, 10)
        self.J_1_EE, self.J_2_EE = round(p.J_1_EE / p.N_E, 10), round(p.J_2_EE / p.N_E, 10)
        self.J_0_IE, self.J_0_II = round(p.J_0_IE / p.N_E, 10), round(p.J_0_II / p.N_I, 10)
        self.J_1_IE, self.J_2_IE = round(p.J_1_IE / p.N_E, 10), round(p.J_2_IE / p.N_E, 10)

    def _initialise_sim_duration(self):
        """Adjust the simulation duration based on the time constant."""
        self.params.sim_duration = round(self.params.sim_duration * self.params.time_constant, 10)

    def _equilibrate(self):
        """
        Bring the model to an equilibrium state before simulation.

        - Runs the simulation without sensory input to find a stable state.
        - Stores the equilibrium state.
        - Stores the number of spontaneously active neurons.
        """
        # If equilibration has already been run, and no params have been updated since
        if self.state.equilibrated:
            self._restore_equilibrium_state()
            return

        # Run the model for equilibration
        self.run()

        # Store the equilibrium state
        self.equilibrium.E, self.equilibrium.I = self.E.copy(), self.I.copy()
        self.equilibrium.x, self.equilibrium.y = self.x.copy(), self.y.copy()

        # Store the index of the first non-spontaneous neuron
        self.spont_index = self.params.N_E - np.count_nonzero(np.max(self.activity.E[-1], axis=0) > 0)
        self.equilibrium.spont_index = self.spont_index

        # Update equilibrated flag
        self.state.equilibrated = True

    def _restore_equilibrium_state(self):
        """Restores stored equilibrium state."""
        self.E, self.I = self.equilibrium.E, self.equilibrium.I
        self.x, self.y = self.equilibrium.x, self.equilibrium.y
        self.activity = Activity()
        self.spont_index = self.equilibrium.spont_index

    def _is_equilibrium_state(self):
        """Checks if the current state matches the stored equilibrium state."""
        return (
                self.state.equilibrated and
                np.array_equal(self.E, self.equilibrium.E) and
                np.array_equal(self.I, self.equilibrium.I) and
                np.array_equal(self.x, self.equilibrium.x) and
                np.array_equal(self.y, self.equilibrium.y)
        )

    def _reset_column_stimuli(self):
        """Resets the stimuli."""
        self.stimuli = Stimuli(self.params.P)

    def update_params(self, **updates):
        """Updates model parameters and re-equilibrates if necessary."""
        for key, value in updates.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            elif key == 'e_E':
                self.params.e_E = value

        non_equilibration_params = {'sim_duration', 'time_constant'}
        if not set(updates.keys()).issubset(non_equilibration_params):
            self.state.equilibrated = False
            self.state.initialising = True

        # Re-initialise the model
        self._initialise_model()

    def add_stimulus(self, col, A, start, stop):
        """Adds a stimulus to a specific column."""
        self.stimuli.add_stimulus(col, A, start, stop)

    def get_sensory_input_matrix(self):
        """Returns a copy of the sensory input matrix."""
        return self.sensory_input_matrix.copy()

    def get_activity(self):
        """Returns a copy of the activity."""
        return self.activity.copy()

    def ode_model(self, t, v):
        """"Defines the ODE system governing the model dynamics."""
        p = self.params

        # Extract population data
        self.unpack_and_store_data(v)

        # Compute sensory input
        s = self.z_func(t)

        # Compute the changes in x and y
        dxdt = ((1 - self.x) / p.tau_rec) - p.U * self.x * self.E
        dydt = ((1 - self.y) / p.tau_rec) - p.U * self.y * self.I

        # Compute the input contributions
        UxE_sum = np.diag(self.E @ (p.U * self.x).T)[:, np.newaxis]
        UyI_sum = np.diag(self.I @ (p.U * self.y).T)[:, np.newaxis]
        E_sum = np.sum(self.E, axis=1, keepdims=True)
        I_sum = np.sum(self.I, axis=1, keepdims=True)

        # Compute lateral interactions using shifted sums
        UxE_1  = shift_array(UxE_sum, -1) + shift_array(UxE_sum, 1)
        UxE_2 = shift_array(UxE_sum, -2) + shift_array(UxE_sum, 2)
        E_sum_1 = shift_array(E_sum, -1) + shift_array(E_sum, 1)
        E_sum_2 = shift_array(E_sum, -2) + shift_array(E_sum, 2)

        # Collect the total input to E and I populations
        input_E = s + self.e_E + (self.J_0_EE * UxE_sum + self.J_0_EI * UyI_sum + self.J_1_EE * UxE_1 + self.J_2_EE * UxE_2)
        input_I = self.e_I + (self.J_0_IE * E_sum + self.J_0_II * I_sum + self.J_1_IE * E_sum_1 + self.J_2_IE * E_sum_2)

        # Compute the changes in E and I populations
        dEdt = (-self.E + (1 - p.tau_E_ref * self.E) * relu(input_E)) / p.tau_E
        dIdt = (-self.I + (1 - p.tau_I_ref * self.I) * relu(input_I)) / p.tau_I

        return self.pack_data(dEdt, dIdt, dxdt, dydt)


    def run(self, all_neurons = False):
        """Runs the model simulation."""
        # We need to re-equilibrate first if the model has progressed out of equilibrium
        if not self.state.initialising and not self._is_equilibrium_state():
            #logger.info("Re-establishing equilibrium before running...")
            self._equilibrate()

        # Set the correct simulation duration
        duration = self.params.sim_duration if self.state.equilibrated else self.params.T_equil

        # Set the correct spontaneously active neurons index
        self.spont_index = 0 if all_neurons else self.spont_index

        # Set the sensory input matrix
        self.set_sensory_input_matrix()

        # Solve the ODE system to get population activities
        result = solve_ivp(self.ode_model, (0, duration), self.pack_data(self.E, self.I, self.x, self.y),
                           t_eval=np.arange(0, duration, self.params.dt), method='RK45')

        self.unpack_and_store_result(result)

        # Clear the stimuli that were added for this particular run
        self._reset_column_stimuli()


    def pack_data(self, E, I, x, y):
        """Flattens and concatenates arrays for ODE solver."""
        return np.concatenate([E.flatten(), I.flatten(), x.flatten(), y.flatten()])


    def unpack_and_store_data(self, arr):
        """Restores flattened state vectors into structured arrays."""
        P, N_E, N_I = self.params.P, self.params.N_E, self.params.N_I
        self.E = arr[: P * N_E].reshape(P, N_E)
        self.I = arr[P * N_E: P * (N_E + N_I)].reshape(P, N_I)
        self.x = arr[P * (N_E + N_I): P * (2 * N_E + N_I)].reshape(P, N_E)
        self.y = arr[P * (2 * N_E + N_I): 2 * P * (N_E + N_I)].reshape(P, N_I)


    def unpack_and_store_result(self, res):
        """Stores population activities and states using the solver output."""
        p = self.params
        arr = res.y
        size = res.t.size

        E_end = p.P * p.N_E
        I_end = E_end + (p.P * p.N_I)
        x_end = I_end + (p.P * p.N_E)
        y_end = x_end + (p.P * p.N_I)

        def reshape_activity(start, end, neurons):
            """
            Reshape a section of the solver output into a 3D array (time x P x neurons).
            """
            return arr[start:end].reshape(p.P, neurons, size).transpose(2, 0, 1)

        self.activity.E = reshape_activity(0, E_end, p.N_E)
        self.activity.I = reshape_activity(E_end, I_end, p.N_I)
        self.activity.x = reshape_activity(I_end, x_end, p.N_E)
        self.activity.y = reshape_activity(x_end, y_end, p.N_I)

        self.E = self.activity.E[-1, :, :]
        self.I = self.activity.I[-1, :, :]
        self.x = self.activity.x[-1, :, :]
        self.y = self.activity.y[-1, :, :]


    def lambda_S(self, M, A):
        """Computes the spatial decay factor λ_S(A) for synaptic input."""
        p = self.params
        delta = np.select(
            [np.arange(p.P) < M, np.arange(p.P) >= M],
            [p.delta_left, p.delta_right]
        )
        return p.lambda_C + np.maximum(0, (A - p.alpha) / delta)


    def h_func(self, Q, A):
        """Computes the spatial component h."""
        # Compute the spatial decay factor
        lambda_s = self.lambda_S(Q, A)

        # Compute exponential exponent
        distance = np.abs(Q - np.arange(self.params.P))

        # Compute the h values
        h_values = (A * np.exp(-distance / lambda_s))[:, np.newaxis]

        # Only apply the spatial component to the spontaneously active neurons
        h = np.tile(h_values, (1, self.params.N_E - self.spont_index))

        return h


    def z_func(self, t):
        """
        Retrieve the sensory input at time `t` from the sensory input matrix.

        The sensory input matrix stores external stimuli over discrete time steps for each column.
        This function retrieves the appropriate sensory input for the current time step.

        Inputs:
            t: Current simulation time.
        """
        # Return zero if there is no sensory input
        if self.state.equilibrated and t <= self.params.sim_duration:
            # Determine the index in the sensory input matrix based on time step
            index = max(0, int(np.ceil(t / self.params.stim_step_size)) - 1)
            return self.sensory_input_matrix[index]
        else:
            return np.zeros((1, 1))


    def set_sensory_input_matrix(self):
        """
        Generate the sensory input matrix based on input stimuli.

        This function constructs a time-series matrix where:
        - Rows correspond to time steps.
        - Columns correspond to spatial positions (P).
        - Values represent the stimulus amplitude over time.

        The stimuli are specified for each column with:
        - `begin`: Start time of the stimulus (relative to simulation duration).
        - `end`: End time of the stimulus.
        - `amplitude`: Strength of the stimulus.
        """
        # Only need to stimuli when we are not equilibrating
        if self.state.equilibrated:
            p = self.params

            # Initialise stimulus matrix (time x columns)
            total_stim_steps = int(round(p.sim_duration / p.stim_step_size))
            self.stim_matrix = np.zeros((p.P, total_stim_steps))

            # Populate stimulus matrix using the added stimuli
            for col in range(p.P):
                for stim_event in self.stimuli.get_stimuli(col):
                    amplitude = stim_event["amplitude"]
                    start_idx = int(total_stim_steps * stim_event["begin"] * p.time_constant / p.sim_duration)
                    end_idx = int(total_stim_steps * stim_event["end"] * p.time_constant / p.sim_duration)

                    self.stim_matrix[col, start_idx:end_idx] += amplitude

            # Add the full sensory input matrix (for the defined discrete time steps)
            self.sensory_input_matrix = np.zeros((total_stim_steps, p.P, p.N_E))
            h = np.zeros((p.P, p.N_E))
            for Q in range(p.P):
                for i, step in enumerate(self.stim_matrix[Q]):
                    if step > 0:
                        h[:, self.spont_index:p.N_E] = self.h_func(Q, step)
                        self.sensory_input_matrix[i] += h

if( __name__ == "__main__"):
    from Params import Params

    # --- Get Model Parameters --- #
    params = Params(sim_duration=0.5)

    # --- Initialise the model --- #
    a1_model = A1Model(params)

    # --- Define the Stimuli --- #
    a1_model.add_stimulus(8, 4, 0, 0.05)
    a1_model.add_stimulus(9, 4, 0, 0.05)

    # --- Run Simulation for Figure 2 --- #
    a1_model.run()
