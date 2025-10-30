# Processing of Sounds by Population Spikes in a Model of A1

This repository contains the Python implementation and analysis scripts accompanying the report which reproduces and extends the auditory cortex model of **Loebel et al. (2007)** — exploring nonlinear auditory response phenomena in a recurrent network model of primary auditory cortex (A1).

---

## Model Overview

The model follows the rate-based cortical network introduced by **Loebel, Nelken, and Tsodyks (2007)**, where each iso-frequency column in A1 is a recurrent excitatory–inhibitory network incorporating
short-term synaptic depression (STD). Excitatory synapses deplete with use and recover over time, producing transient, synchronous bursts of firing known as Population Spikes (PS).

These population-level dynamics explain several nonlinear auditory phenomena:
- Forward masking and its recovery dynamics
- Hypersensitive locking suppression
- Frequency tuning curve (FTC) shaping
- PS propagation and sweep direction selectivity
- Signal-to-noise ratio (SNR) effects
- Stimulus-specific adaptation (SSA)

---

## Running the Code

To run specific analyses individually:

```bash
python scripts/forward_masking.py
python scripts/ssa.py
python scripts/snr.py
```

---

## Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Reference

Loebel A., Nelken I., & Tsodyks M. (2007).
*Processing of sounds by population spikes in a model of primary auditory cortex (A1).*
**Frontiers in Neuroscience, 1(1), 15–25.**
[https://doi.org/10.3389/neuro.01.1.1.015.2007](https://doi.org/10.3389/neuro.01.1.1.015.2007)

---
