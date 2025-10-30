from dataclasses import dataclass


@dataclass
class Params:
    N_E: int = 100
    N_I: int = 100
    tau_E: float = 0.001
    tau_I: float = 0.001
    tau_E_ref: float = 0.003
    tau_I_ref: float = 0.003
    tau_rec: float = 0.8
    U: float = 0.5
    J_0_EE: float = 6
    J_0_EI: float = -4
    J_0_IE: float = 0.5
    J_0_II: float = -0.5
    J_1_EE: float = 0.045
    J_2_EE: float = 0.015
    J_1_IE: float = 0.0035
    J_2_IE: float = 0.0015
    e_E_1: int = -10
    e_I_1: int = -10
    e_E_NE: int = 10
    e_I_NI: int = 10
    P: int = 15
    lambda_C: float = 0.25
    alpha: int = 2
    delta_left: int = 5
    delta_right: int = 5
    T_equil: float = 5
    time_constant: float = 1
    dt: float = 0.001
    stim_step_size: float = 0.001
    seed: int = 47
    initial_x: float = 0.25
    initial_y: float = 0.25
    sim_duration: float = 2
    delta_e: float = 0