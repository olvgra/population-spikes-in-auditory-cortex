# A1Model: Model of the Primary Auditory Cortex

## Overview
The `A1Model` class houses all logic for implementing the A1 model (Loebel et al. 2007). 

## Features
- Initialises the model and brings it to a state of equilibrium.
- Supports the addition of external stimuli.
- Ensures that the system always starts a new run from the equilibrium state.
- Allows parameter updates with automatic re-equilibration.
- Uses `solve_ivp` for numerical integration.

## Dependencies
Ensure the following dependencies are installed before using the model:
```bash
pip install numpy scipy
```

## Usage

### 1. Initialise the Model
```python
from classes.A1Model import A1Model
from classes.Params import Params

"""
Default parameters are defined in the Params class.
Add any specific parameter updates as arguments.
""" 
model_params = Params(sim_duration=1)
model = A1Model(model_params)
```

### 2. Run the Model Simulation
```python
model.run()
```

### 3. Add Stimuli
```python
"""
By default, the model adds external stimuli to the spontaneously active neurons only.
Set the `all_neurons` flag to `True` to apply the stimulus to all the neurons.
""" 
model.add_stimulus(col=2, A=0.5, start=0.1, stop=0.5)

# Run model with stimulus added only to spont. active neurons
model.run()

# Run model with stimulus added only to ALL neurons
model.run(all_neurons=True)
```

### 4. Retrieve Simulation Results
```python
activity = model.get_activity()
print(activity.E)  # Excitatory activity over time
print(activity.I)  # Inhibitory activity over time
```

### 5. Update Model Parameters
```python
# Updating the parameters will ensure a full re-equilibration happens
model.update_params(N_E=10, J_0_EE=0.2)
```


## Class Breakdown
### `A1Model`
- **Initialisation (`__init__`)**: Initialises neural populations, parameters, equilibrium state, and activity tracking.
- **Equilibration (`_equilibrate`)**: Runs the model without input to find a stable state.
- **Simulation (`run`)**: Solves the differential equations governing population dynamics using an ODE solver.
- **Stimulus Adding (`add_stimulus`)**: Add external stimuli to the excitatory neurons.
- **Parameter Updates (`update_params`)**: Updates parameters and re-equilibrates if necessary.
- **Retrieval Functions (`get_activity`, `get_sensory_input_matrix`)**: Returns data relating the current model run.


