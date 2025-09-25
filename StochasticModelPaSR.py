#%% Stochastic Modeling of PaSR by YY
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import time
import csv
from StochasticFunctions import *
from math import floor
import yaml
import os

# === Create folders ===
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# === Read input ===
with open("input.yaml", "r") as f:
    params = yaml.safe_load(f)

omega_turb_list = params["Mixing frequency [Hz]"]
if not isinstance(omega_turb_list, list):
    omega_turb_list = [float(omega_turb_list)]
else:
    omega_turb_list = [float(x) for x in omega_turb_list]
niter = int(params["Number of iteration"])
Np = int(params["Number of particles"])
T_unburnt = float(params["Unburnt temperature [K]"])
p0 = getattr(ct, params["Unburnt pressure [Pa]"])
C_phi = float(params["Phi constant for IEM"])
dt = float(params["Time interval"])
flame_type = params["Flame type [nonpremixed / premixed]"]
mixing_model_type = params["Mixing model [KerM / MC / IEM]"]
mech = params["Mechanism path"]
sigma_k = float(params["Sigma_k constant for KerM"])
reaction_method = params["Reaction method [LFR / ODE]"]

fuel = params["fuel"]
oxidizer = params["oxidizer"]

# === Cantera ===
gas = ct.Solution(mech)

# === Initial state ===
Psi_unburnt, gas = get_unburnt_state(gas, T_unburnt, p0, fuel, oxidizer, flame_type)
Psi_particles = get_unburnt_state_equilibrium_particles(mech, gas, T_unburnt, p0, Np, fuel, oxidizer)

mixing_model_parameter = {"C_phi": C_phi, "sigma_k": sigma_k}
mw = gas.molecular_weights

for omega_turb in omega_turb_list:
    print(f"=== Running simulation for mixing frequency = {omega_turb} Hz ===")
    residence_time = 1 / omega_turb
    N_replace = floor(Np * dt / residence_time)
    nx, ny = Psi_particles.shape
    TT_particles = np.zeros(Np)
    iterations = np.arange(1, niter + 1)
    times = (iterations - 1) * dt
    start_time = time.time()

    # === Iteration ===
    for n in range(niter):
        progress = (n + 1) / niter * 100
        print(f"\rProgress: {progress:.1f}%", end="")
        
        # 1. Flow
        Psi_particles = through_flow_process(flame_type, Psi_particles, Psi_unburnt, N_replace, mixing_model_type)

        # 2. Mixing
        if mixing_model_type == 'IEM':
            Psi_particles = evolution_mixing_process_IEM(Psi_particles, mixing_model_parameter, omega_turb, dt)
        elif mixing_model_type == 'MC':
            Psi_particles = evolution_mixing_process_MC(Psi_particles, omega_turb, dt)
        elif mixing_model_type == 'KerM':
            Psi_particles = evolution_mixing_process_KerM(Psi_particles, mixing_model_parameter, omega_turb, dt, gas, fuel, oxidizer)
        else:
            raise ValueError(f"Unknown mixing model: {mixing_model_type}")

        # 3. Reaction
        if reaction_method == "LFR":
            Psi_particles, TT, R_all, HRR_all, rho_all = evolution_reaction_process_LFR(gas, Psi_particles, dt)
            output_filename = f"data/ZT_w_{omega_turb:.0f}Hz_LFR.csv"
            fig_filename    = f"figures/ZT_w_{omega_turb:.0f}Hz_LFR.png"
        elif reaction_method == "ODE":
            Psi_particles, TT, R_all, HRR_all, rho_all = evolution_reaction_process_ODE(gas, Psi_particles, dt)
            output_filename = f"data/ZT_w_{omega_turb:.0f}Hz_ODE.csv"
            fig_filename    = f"figures/ZT_w_{omega_turb:.0f}Hz_ODE.png"
        else:
            raise ValueError(f"Unknown reaction method: {reaction_method}")

        # 4. Output
        R_all_array = np.array(R_all)
        HRR_all_array = np.array(HRR_all)
        Rho_all_array = np.array(rho_all)

        for k in range(Np):
            YYY = Psi_particles[2:, k] / mw
        TT_particles = TT

        Y_fuel = Psi_unburnt['fuel'][2]
        Y_oxidizer = Psi_unburnt['oxidizer'][2]

        Z_all = np.zeros(Np)
        for i in range(Np):
            Y_i = Psi_particles[2:, i]
            Z_all[i] = compute_mixture_fraction_direct(Y_i, Y_fuel, Y_oxidizer)

        species_names = gas.species_names
        if n >= niter - 5:
            if n == niter - 5:
                with open(output_filename, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['time', 'particle_id', 'T', 'Z', 'P', 'Rho', 'HRR', 'R'] + [f'Y_{sp}' for sp in species_names]
                    writer.writerow(header)

            with open(output_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                for i in range(Np):
                    T_i = TT_particles[i]
                    Z_i = Z_all[i]
                    P_i = gas.P
                    HRR_i = HRR_all_array[i]
                    R_i = R_all_array[i, :]
                    Y_i = Psi_particles[2:, i]
                    rho_i = Rho_all_array[i]
                    row = [times[n], i, T_i, Z_i, P_i, rho_i, HRR_i, R_i] + list(Y_i)
                    writer.writerow(row)

        # 6. Time
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTime: {elapsed_time:.3f} s")
        
    # === Final figure ===
    plt.figure(figsize=(5, 4))
    plt.plot(Z_all, TT_particles, 'k.', ms=0.8, alpha=0.8)
    plt.xlim([0, 1.05])
    plt.ylim([0, 2500])
    plt.xlabel("Z")
    plt.ylabel("T")
    plt.title(f"w={omega_turb:.0f} Hz")
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    plt.close()
print("Simulation finished.")