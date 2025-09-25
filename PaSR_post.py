#%%
import numpy as np
import pandas as pd
import cantera as ct
import yaml
import os
import matplotlib.pyplot as plt
import PyCSP.Functions as csp
from StochasticFunctions import *

# === Create folders ===
os.makedirs("data_for_PaSR", exist_ok=True)

# === Read input ===
with open("input.yaml", "r") as f:
    params = yaml.safe_load(f)

omega_turb_list = params["Mixing frequency [Hz]"]
if not isinstance(omega_turb_list, list):
    omega_turb_list = [float(omega_turb_list)]
else:
    omega_turb_list = [float(x) for x in omega_turb_list]
omega_turb_list = np.array(omega_turb_list)
tau_m_list = 1 / omega_turb_list

mech = params["Mechanism path"]
gas = ct.Solution(mech)
fuel = params["fuel"]
oxidizer = params["oxidizer"]

reaction_methods = ["LFR", "ODE"]

data_all, results = load_simulation_data(omega_turb_list)

T_all     = results["T_all"]
Z_all     = results["Z_all"]
P_all     = results["P_all"]
HRR_all   = results["HRR_all"]
R_all     = results["R_all"]
rho_all   = results["rho_all"]
Y_all     = results["Y_all"]

T_all_lfr   = results["T_all_lfr"]
Z_all_lfr   = results["Z_all_lfr"]
P_all_lfr   = results["P_all_lfr"]
HRR_all_lfr = results["HRR_all_lfr"]
R_all_lfr   = results["R_all_lfr"]
rho_all_lfr = results["rho_all_lfr"]
Y_all_lfr   = results["Y_all_lfr"]

particles = results["particles"]

# === LFR error ===
diff_LFR = np.abs(HRR_all - HRR_all_lfr)
error_LFR = np.sum(diff_LFR)

# === SFR & FFR ===
(error_SFR, error_FFR, Da_SFR, Da_FFR, gamma_SFR, gamma_FFR, diff_SFR, diff_FFR) = compute_sfr_ffr_error(Y_all_lfr,R_all_lfr,HRR_all_lfr,HRR_all,tau_m_list,particles)

# === Chomiak ===
error_chomiak, Da_chomiak, gamma_chomiak, diff_chomiak = compute_chomiak_error(gas,params,Y_all_lfr,R_all_lfr,rho_all_lfr,HRR_all_lfr,HRR_all,tau_m_list,particles)

# # === Jacobian ===
gas_csp = csp.CanteraCSP(mech)

error_jacobian, Da_jacobian, gamma_jacobian, diff_jacobian, tau_c_jacobian = compute_jacobian_error(gas_csp,T_all_lfr,P_all_lfr,Y_all_lfr,HRR_all_lfr,HRR_all,tau_m_list,particles)

# === Best model per point (Most optimal error) ===
(minimum_error_pointwise,minimum_diff_pointwise,minimum_diff_pointwise_index,X,idx) = compute_best_model(Z_all,T_all,Y_all,diff_chomiak,diff_SFR,diff_FFR,diff_jacobian)
