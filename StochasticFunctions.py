import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from scipy.integrate import solve_ivp
import os
import PyCSP.Functions as csp
import pandas as pd

def compute_mixture_fraction_direct(Y, Y_fuel, Y_oxidizer):
    numerator = np.dot(Y - Y_oxidizer, Y_fuel - Y_oxidizer)
    denominator = np.dot(Y_fuel - Y_oxidizer, Y_fuel - Y_oxidizer)
    
    if denominator == 0:
        return 0.0
    Z = numerator / denominator
    return np.clip(Z, 0.0, 1.0)

def get_unburnt_state_equilibrium_particles(mech, gas, T_unburnt, p0, Np, fuel, oxidizer):
    gas_ref = ct.Solution(mech)
    gas_ref.TP = T_unburnt, p0
    gas_ref.set_equivalence_ratio(1.0, fuel, oxidizer)
    gas_ref.equilibrate('HP')
    Zst = gas_ref.mixture_fraction(fuel, oxidizer)
    mw = gas.molecular_weights
    Psi_list = []

    Nst = int(Np * 0.6)

    for _ in range(Nst):
        g = ct.Solution(mech)
        g.TP = T_unburnt, p0
        g.set_equivalence_ratio(1.0, fuel, oxidizer)
        g.equilibrate('HP')
        h = g.enthalpy_mass
        p = g.P
        Y = g.Y
        Psi = np.concatenate(([h], [p], Y))
        Psi_list.append(Psi)

    N_random = Np - Nst
    for i in range(N_random):
        Z = (i + 1) / (N_random + 1)
        phi = (Z / (1 - Z)) * ((1 - Zst) / Zst + 1e-12)

        try:
            g = ct.Solution(mech)
            g.TP = T_unburnt, p0
            g.set_equivalence_ratio(phi, fuel, oxidizer)
            g.equilibrate('HP')
            h = g.enthalpy_mass
            p = g.P
            Y = g.Y
            Psi = np.concatenate(([h], [p], Y))
            Psi_list.append(Psi)
        except Exception as e:
            print(f"Z={Z:.3f} not calculated: {e}")

    if len(Psi_list) != Np:
        print(f"Warning!")

    Psi_particles = np.array(Psi_list).T
    return Psi_particles

def get_unburnt_state(gas, T_unburnt, p0, fuel, oxidizer, flame_type):
    nsp = gas.n_species
    Phi = 1
    
    # io2 = gas.species_index('O2') #Cabra
    # in2 = gas.species_index('N2') #Cabra
    # ih2 = gas.species_index('H2') #Cabra
    # ih2o = gas.species_index('H2O') #Cabra

    # === Fuel ve oxidizer ===
    all_species = {**fuel, **oxidizer}
    species_indices = {sp: gas.species_index(sp) for sp in all_species.keys()}
    
    if flame_type == 'premixed':
        gas.set_equivalence_ratio(Phi, fuel, oxidizer)
        gas.TP = T_unburnt, p0
        enthalpy = gas.enthalpy_mass
        pressure = gas.P
        mass_fractions = gas.Y
        Psi_unburnt = np.hstack([enthalpy, pressure, mass_fractions])

    elif flame_type == 'nonpremixed':
        Psi_unburnt = {}

        # === FUEL: Saf H2 ===
        X_fuel = np.zeros(nsp)
        for sp, val in fuel.items():
            X_fuel[species_indices[sp]] = val
        # X_fuel[ih2] = fuel["H2"] #Cabra
        # X_fuel[in2] = fuel["N2"] #Cabra
        gas.TPX = T_unburnt, p0, X_fuel
        Psi_unburnt['fuel'] = [gas.enthalpy_mass, gas.P, gas.Y]

        # === OXIDIZER: Air scaled by 1/Phi ===
        lambda_ratio = 1.0 / Phi
        X_oxidizer = np.zeros(nsp)
        # X_oxidizer[io2] = oxidizer["O2"] #Cabra
        # X_oxidizer[in2] = oxidizer["N2"] #Cabra
        # X_oxidizer[ih2o] = oxidizer["H2O"] #Cabra
        for sp, val in oxidizer.items():
            X_oxidizer[species_indices[sp]] = val
        X_oxidizer /= np.sum(X_oxidizer)

        gas.TPX = T_unburnt, p0, X_oxidizer
        Psi_unburnt['oxidizer'] = [gas.enthalpy_mass, gas.P, gas.Y]

    return Psi_unburnt, gas

def get_equilibrium_state(gas):
        gas.equilibrate('HP')
        hmass_eq = gas.enthalpy_mass  # Specific enthalpy [J/kg]
        T_eq = gas.T                  # Temperature [K]
        P_eq = gas.P                  # Pressure [Pa]
        Y_eq = gas.Y                  # Mass fractions [-]
        Psi_eq = np.concatenate(([hmass_eq], [P_eq], Y_eq))
        return Psi_eq, T_eq

def evolution_mixing_process_MC(Psi_particles, omega_turb, dt):
    nx, Np = Psi_particles.shape
    omdt = omega_turb * dt

    nmix = int(1.5 * omdt * Np + 1)
    pmix = 1.5 * omdt * Np / nmix

    for _ in range(nmix):
        p = int(np.floor(np.random.rand() * Np))
        q = int(np.floor(np.random.rand() * Np))
        if np.random.rand() < pmix:
            a = np.random.rand()
            phi_pq = 0.5 * (Psi_particles[:, p] + Psi_particles[:, q])
            Psi_particles[:, p] += -a * (Psi_particles[:, p] - phi_pq)
            Psi_particles[:, q] += -a * (Psi_particles[:, q] - phi_pq)

    return Psi_particles

def evolution_mixing_process_IEM(Psi_particles, mixing_model_parameter, omega_turb, dt):
        C_phi = mixing_model_parameter['C_phi']
        Np = Psi_particles.shape[1]
        Psi_mean = np.mean(Psi_particles, axis=1, keepdims=True)
        for i in range(Np):
            Psi_particles[:, i] = Psi_mean[:, 0] + \
                (Psi_particles[:, i] - Psi_mean[:, 0]) * np.exp(-0.5 * C_phi * omega_turb * dt)
        return Psi_particles

def evolution_mixing_process_KerM(Psi_particles, mixing_model_parameter, omega_turb, dt, gas, fuel, oxidizer):
    sigma_k = mixing_model_parameter.get('sigma_k', 0.05)
    if sigma_k < 0.01:
        print("Warning: sigma_k is recommended to be in [0.01 ~ inf]")

    Np = Psi_particles.shape[1]

    Zs = np.zeros(Np)
    for i in range(Np):
        gas.Y = Psi_particles[2:, i]
        Zs[i] = gas.mixture_fraction(fuel, oxidizer)

    Ms = np.ones(Np) / Np

    # CDF
    sorted_id = np.argsort(Zs)
    CDF = np.zeros_like(Ms)
    CDF[sorted_id] = np.cumsum(Ms[sorted_id])

    varZ = np.var(Zs)
    dvar = 0
    Nc = int(Np * max(0.1 / sigma_k, 1))

    for _ in range(Nc):
        p = np.random.randint(Np)
        q = np.random.randint(Np)
        d = CDF[p] - CDF[q]
        f = np.exp(-d**2 / sigma_k**2 / 4)
        dvar += 0.5 * f * (Zs[p] - Zs[q])**2 / Nc

    coeff = varZ / dvar if dvar != 0 else 0.0

    omdt = omega_turb * dt
    nmix = int(1.5 * omdt * Np * coeff + 1)
    pmix = 1.5 * omdt * Np * coeff / nmix

    for _ in range(nmix):
        p = np.random.randint(Np)
        q = np.random.randint(Np)
        d = CDF[p] - CDF[q]
        f = np.exp(-d**2 / sigma_k**2 / 4)
        if np.random.rand() < f and np.random.rand() < pmix:
            a = np.random.rand()
            phi_pq = (Psi_particles[:, p] + Psi_particles[:, q]) / 2
            Psi_particles[:, p] += -a * (Psi_particles[:, p] - phi_pq)
            Psi_particles[:, q] += -a * (Psi_particles[:, q] - phi_pq)

    return Psi_particles

def evolution_reaction_process_ODE(gas, Psi_particles, dt):
        Np = Psi_particles.shape[1]
        Psi_particles_new = np.zeros_like(Psi_particles)
        Temperature_particles = np.zeros(Np)
        R_all = []
        HRR_all = []
        rho_all = []
        for i in range(Np):
            Psi_i = Psi_particles[:, i]
            hmass_i, P_i = Psi_i[0], Psi_i[1]
            Y_i = Psi_i[2:]
            gas.HPY = hmass_i, P_i, Y_i
            
            Psi_0 = np.concatenate(([gas.T], [P_i], Y_i))
            reaction_stats = []

            def conhp(t, Psi_0, gas):
                return conhp_rhs(t, Psi_0, gas, reaction_stats)
                    
            sol = solve_ivp(conhp, [0, dt], Psi_0, args=(gas,), method='BDF', rtol=1e-8, atol=1e-10)

            R_particle = reaction_stats[-1][1]
            HRR_particle = reaction_stats[-1][2]

            R_all.append(R_particle)
            HRR_all.append(HRR_particle)

            gas.TPY = sol.y[0, -1], sol.y[1, -1], sol.y[2:, -1]
            rho_particle = gas.density
            rho_all.append(rho_particle)

            Psi_particles_new[:, i] = np.concatenate(([gas.enthalpy_mass], [gas.P], gas.Y))
            Temperature_particles[i] = gas.T
        
        return Psi_particles_new, Temperature_particles, R_all, HRR_all, rho_all

def evolution_reaction_process_LFR(gas, Psi_particles, dt):
    Np = Psi_particles.shape[1]
    Psi_particles_new = np.zeros_like(Psi_particles)
    Temperature_particles = np.zeros(Np)
    R_all = []
    HRR_all = []
    rho_all = []

    for i in range(Np):
        Psi_i = Psi_particles[:, i]
        hmass_i, P_i = Psi_i[0], Psi_i[1]
        Y_i = Psi_i[2:]

        gas.HPY = hmass_i, P_i, Y_i
        T_i = gas.T
        rho_i = gas.density
        particle_mass = 0.1
        volume = particle_mass / rho_i

        reac = ct.IdealGasConstPressureReactor(gas, volume=volume)
        netw = ct.ReactorNet([reac])
        t0 = netw.time
        netw.advance(t0 + dt)

        R = gas.net_production_rates * gas.molecular_weights  # [kg/m^3/s]
        HRR = gas.heat_release_rate  # [W/m^3]

        Psi_particles_new[:, i] = np.concatenate(([gas.enthalpy_mass], [gas.P], gas.Y))
        Temperature_particles[i] = gas.T
        rho_particle = gas.density

        R_all.append(R)
        HRR_all.append(HRR)
        rho_all.append(rho_particle)

    return Psi_particles_new, Temperature_particles, R_all, HRR_all, rho_all

def conhp_rhs(t, Psi_0, gas, reaction_stats):
    gas.TPY = Psi_0[0], Psi_0[1], Psi_0[2:]

    omega_dot = gas.net_production_rates
    T_dot = -gas.T * ct.gas_constant * np.dot(gas.standard_enthalpies_RT, omega_dot) / (gas.density * gas.cp_mass)
    R_vector = gas.standard_enthalpies_RT * omega_dot
    HRR = - (gas.density * gas.cp_mass) * T_dot

    reaction_stats.append((t, R_vector, HRR))

    P_dot = 0
    dYdt = reaction_rate(gas)

    return np.concatenate(([T_dot], [P_dot], dYdt))

def reaction_rate(gas):
        mw = gas.molecular_weights
        omega_dot = gas.net_production_rates
        rrho = 1.0 / gas.density
        return rrho * (mw * omega_dot)

def through_flow_process(flame_type, Psi_particles, Psi_unburnt, N_replace, mixing_model_type):
        Np = Psi_particles.shape[1]

        ip_particle_replace = np.random.permutation(Np)[:N_replace]
        if flame_type == 'premixed':
            for i in range(N_replace):
                Psi_particles[:, ip_particle_replace[i]] = Psi_unburnt
        elif flame_type == 'nonpremixed':
            if N_replace == 1:
                if np.random.rand() < 0.5:
                    Psi_particles[:, ip_particle_replace[0]] = np.hstack(Psi_unburnt['oxidizer'])

                else:
                    Psi_particles[:, ip_particle_replace[0]] = np.hstack(Psi_unburnt['fuel'])

            else:
                N_replace_oxidizer = round(0.5 * N_replace)
                
                for i in range(N_replace_oxidizer):
                    Psi_particles[:, ip_particle_replace[i]] = np.hstack(Psi_unburnt['oxidizer'])

                for i in range(N_replace_oxidizer, N_replace):
                    Psi_particles[:, ip_particle_replace[i]] = np.hstack(Psi_unburnt['fuel'])
            
        return Psi_particles
    

def load_simulation_data(omega_turb_list, base_folder="data"):
    data_all = []

    for omega in omega_turb_list:
        files = {
            "LFR": os.path.join(base_folder, f"ZT_w_{omega:.0f}Hz_LFR.csv"),
            "ODE": os.path.join(base_folder, f"ZT_w_{omega:.0f}Hz_ODE.csv")
        }
        entry = {"omega_turb": omega}
        for method, filename in files.items():
            if not os.path.exists(filename):
                continue
            df = pd.read_csv(filename)
            last_time = df['time'].max()
            df_last = df[df['time'] == last_time].copy()

            HRR_array = -df_last['HRR'].to_numpy()
            R_array_strings = df_last['R'].to_numpy()
            R_arrays = np.array([np.fromstring(s.strip('[]'), sep=' ') 
                                 for s in R_array_strings])
            T_array = df_last['T'].to_numpy()
            Z_array = df_last['Z'].to_numpy()
            P_array = df_last['P'].to_numpy()
            rho_array = df_last['Rho'].to_numpy()
            Y_columns = [col for col in df_last.columns if col.startswith('Y_')]
            Y_data = df_last[Y_columns].to_numpy()

            entry[f"T_{method.lower()}"] = T_array
            entry[f"Z_{method.lower()}"] = Z_array
            entry[f"P_{method.lower()}"] = P_array
            entry[f"HRR_{method.lower()}"] = HRR_array
            entry[f"R_{method.lower()}"] = R_arrays
            entry[f"Rho_{method.lower()}"] = rho_array
            entry[f"Y_{method.lower()}"] = Y_data
            entry["Y_columns"] = Y_columns
        data_all.append(entry)

    # === Concatenate all frequencies ===
    results = {
        "T_all":     np.concatenate([d["T_ode"] for d in data_all if "T_ode" in d]),
        "Z_all":     np.concatenate([d["Z_ode"] for d in data_all if "Z_ode" in d]),
        "P_all":     np.concatenate([d["P_ode"] for d in data_all if "P_ode" in d]),
        "HRR_all":   np.concatenate([d["HRR_ode"] for d in data_all if "HRR_ode" in d]),
        "R_all":     np.concatenate([d["R_ode"] for d in data_all if "R_ode" in d]),
        "rho_all":   np.concatenate([d["Rho_ode"] for d in data_all if "Rho_ode" in d]),
        "Y_all":     np.concatenate([d["Y_ode"] for d in data_all if "Y_ode" in d]),

        "T_all_lfr":   np.concatenate([d["T_lfr"] for d in data_all if "T_lfr" in d]),
        "Z_all_lfr":   np.concatenate([d["Z_lfr"] for d in data_all if "Z_lfr" in d]),
        "P_all_lfr":   np.concatenate([d["P_lfr"] for d in data_all if "P_lfr" in d]),
        "HRR_all_lfr": -np.concatenate([d["HRR_lfr"] for d in data_all if "HRR_lfr" in d]),
        "R_all_lfr":   np.concatenate([d["R_lfr"] for d in data_all if "R_lfr" in d]),
        "rho_all_lfr": np.concatenate([d["Rho_lfr"] for d in data_all if "Rho_lfr" in d]),
        "Y_all_lfr":   np.concatenate([d["Y_lfr"] for d in data_all if "Y_lfr" in d]),
    }

    results["particles"] = results["Y_all_lfr"].shape[0] // len(omega_turb_list)

    return data_all, results


def compute_sfr_ffr_error(
    Y_all_lfr: np.ndarray,
    R_all_lfr: np.ndarray,
    HRR_all_lfr: np.ndarray,
    HRR_all: np.ndarray,
    tau_m_list: np.ndarray,
    particles: int,
    epsilon: float = 1e-20,
    inf_time: float = 1e20,
    dormant_species: int = 8
) -> tuple:
    tau_c_SFR = np.zeros(particles * len(tau_m_list))
    tau_c_FFR = np.full(particles * len(tau_m_list), inf_time)

    R_org_dormant = R_all_lfr[:, :dormant_species]
    all_Y_dormant = Y_all_lfr[:, :dormant_species]

    for i in range(R_org_dormant.shape[1]):
        Y = all_Y_dormant[:, i]
        R = R_org_dormant[:, i]
        tau_2 = np.abs(Y / R)
        idx = np.abs(R) < epsilon
        tau_2[idx] = -inf_time
        tau_c_SFR = np.maximum(tau_c_SFR, tau_2)
        tau_2[idx] = inf_time
        tau_c_FFR = np.minimum(tau_c_FFR, tau_2)

    diff_SFR, gamma_SFR, Da_SFR = [], [], []
    diff_FFR, gamma_FFR, Da_FFR = [], [], []

    for i in range(HRR_all.shape[0]):
        idx = i // particles
        # --- SFR ---
        Da_sfr = tau_m_list[idx] / tau_c_SFR[i]
        gam_sfr = 1 / (1 + Da_sfr)
        HRR_sfr = HRR_all_lfr[i] * gam_sfr
        diff_SFR.append(np.abs(HRR_sfr - HRR_all[i]))
        Da_SFR.append(Da_sfr)
        gamma_SFR.append(gam_sfr)

        # --- FFR ---
        Da_ffr = tau_m_list[idx] / tau_c_FFR[i]
        gam_ffr = 1 / (1 + Da_ffr)
        HRR_ffr = HRR_all_lfr[i] * gam_ffr
        diff_FFR.append(np.abs(HRR_ffr - HRR_all[i]))
        Da_FFR.append(Da_ffr)
        gamma_FFR.append(gam_ffr)

    error_SFR = np.sum(diff_SFR)
    error_FFR = np.sum(diff_FFR)

    return (
        error_SFR, error_FFR,
        Da_SFR, Da_FFR,
        gamma_SFR, gamma_FFR,
        diff_SFR, diff_FFR
    )
    

def compute_chomiak_error(
    gas: ct.Solution,
    params: dict,
    Y_all_lfr: np.ndarray,
    R_all_lfr: np.ndarray,
    rho_all_lfr: np.ndarray,
    HRR_all_lfr: np.ndarray,
    HRR_all: np.ndarray,
    tau_m_list: np.ndarray,
    particles: int,
) -> tuple[float, list, list, list]:
    species_names = gas.species_names
    fuel_species = [s for s in params["fuel"].keys() if s.upper() != "N2"]
    oxidizer_species = [s for s in params["oxidizer"].keys() if s.upper() != "N2"]

    fuel_idx = [species_names.index(s) for s in fuel_species]
    ox_idx   = [species_names.index(s) for s in oxidizer_species]

    Y_fuel = np.sum(Y_all_lfr[:, fuel_idx], axis=1)
    R_fuel = np.sum(R_all_lfr[:, fuel_idx], axis=1)
    Y_ox   = np.sum(Y_all_lfr[:, ox_idx], axis=1)
    R_ox   = np.sum(R_all_lfr[:, ox_idx], axis=1)

    tau_fuel = -R_fuel / Y_fuel
    tau_ox   = -R_ox / Y_ox

    tau_c_chomiak = np.zeros_like(HRR_all_lfr)
    for i in range(particles):
        tau_c_chomiak[i] = 1 / (np.max([tau_fuel[i], tau_ox[i]]) / rho_all_lfr[i])
    tau_c_chomiak = np.nan_to_num(tau_c_chomiak, nan=1e20)

    diff_chomiak, gamma_chomiak, Da_chomiak = [], [], []
    for i in range(HRR_all.shape[0]):
        idx = i // particles
        Da_ch = tau_m_list[idx] / tau_c_chomiak[i]
        gam_ch = 1 / (1 + Da_ch)
        HRR_ch = HRR_all_lfr[i] * gam_ch
        diff_chomiak.append(np.abs(HRR_ch - HRR_all[i]))
        Da_chomiak.append(Da_ch)
        gamma_chomiak.append(gam_ch)

    error_chomiak = np.sum(diff_chomiak)

    return error_chomiak, Da_chomiak, gamma_chomiak, diff_chomiak


def compute_jacobian_error(
    gas_csp,
    T_all_lfr: np.ndarray,
    P_all_lfr: np.ndarray,
    Y_all_lfr: np.ndarray,
    HRR_all_lfr: np.ndarray,
    HRR_all: np.ndarray,
    tau_m_list: np.ndarray,
    particles: int,
    ne: int = 3,
    tauChem_threshold: float = 1e16
) -> tuple:
    def search_conservative_modes(lambda_real, lambda_mod, ne):
        sorted_indices = np.argsort(lambda_mod)
        elements_to_remove = sorted(sorted_indices[:ne], reverse=True)
        lambda_real_cleaned = list(lambda_real)
        for idx in elements_to_remove:
            del lambda_real_cleaned[idx]
        lambda_real_cleaned = np.abs(lambda_real_cleaned)
        return np.array(lambda_real_cleaned)

    tau_c_jacobian = np.zeros_like(T_all_lfr)

    for p in range(T_all_lfr.shape[0]):
        gas_csp.TPY = T_all_lfr[p], P_all_lfr[p], Y_all_lfr[p]
        gas_csp.constP = P_all_lfr[p]
        gas_csp.jacobiantype = "kinetic"

        lam, R, L, f = gas_csp.get_kernel()
        lambda_real = lam.real
        lambda_imag = lam.imag
        lambda_mod = np.sqrt(lambda_real**2 + lambda_imag**2)

        cleaned_lambdas = search_conservative_modes(lambda_real, lambda_mod, ne)

        if cleaned_lambdas.size > 0:
            minimum_lambda = np.min(cleaned_lambdas)
            tau_chem = 1.0 / minimum_lambda
        else:
            tau_chem = tauChem_threshold

        tau_c_jacobian[p] = tau_chem

    diff_jacobian = np.zeros_like(T_all_lfr)
    gamma_jacobian = np.zeros_like(T_all_lfr)
    Da_jacobian = np.zeros_like(T_all_lfr)

    for i in range(T_all_lfr.shape[0]):
        idx = i // particles
        Da_jacobian[i] = tau_m_list[idx] / tau_c_jacobian[i]
        gamma_jacobian[i] = 1 / (1 + Da_jacobian[i])
        HRR_jacobian = HRR_all_lfr[i] * gamma_jacobian[i]
        diff_jacobian[i] = np.abs(HRR_jacobian - HRR_all[i])

    error_jacobian = np.sum(diff_jacobian)

    return error_jacobian, Da_jacobian, gamma_jacobian, diff_jacobian, tau_c_jacobian


def compute_best_model(
    Z_all: np.ndarray,
    T_all: np.ndarray,
    Y_all: np.ndarray,
    diff_chomiak: list,
    diff_SFR: list,
    diff_FFR: list,
    diff_jacobian: np.ndarray,
    names: list = ["Ch.", "SFR", "FFR", "Jac."],
    color_array: list = ['goldenrod', 'teal', 'darkslateblue', 'salmon'],
    save_path: str = "best_points.png",
    data_folder: str = "data_for_PaSR"
) -> tuple:
    # === Combine errors ===
    diff_pointwise = np.vstack([
        np.array(diff_chomiak),
        np.array(diff_SFR),
        np.array(diff_FFR),
        np.array(diff_jacobian)
    ]).T

    diff_pointwise = np.abs(diff_pointwise)
    minimum_diff_pointwise = np.min(diff_pointwise, axis=1)
    minimum_diff_pointwise_index = np.argmin(diff_pointwise, axis=1)
    minimum_error_pointwise = np.sum(minimum_diff_pointwise)

    # === Plot ===
    plt.figure(figsize=(6, 4))
    for idx_m, name in enumerate(names):
        mask = minimum_diff_pointwise_index == idx_m
        if np.any(mask):
            plt.scatter(Z_all[mask], T_all[mask],
                        color=color_array[idx_m], label=name, s=8, alpha=0.5)

    plt.xlabel("Z [-]")
    plt.ylabel("T [K]")
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([300, 2500])
    plt.yticks([500, 1000, 1500, 2000, 2500])
    plt.tight_layout()

    plt.savefig(save_path, dpi=400, transparent=False, bbox_inches="tight")
    plt.close()

    # === Save data ===
    T_reshaped = T_all.reshape(-1, 1)
    TY_state = np.hstack((T_reshaped, Y_all))
    idx = minimum_diff_pointwise_index.astype(int)

    os.makedirs(data_folder, exist_ok=True)
    np.savetxt(f"{data_folder}/X.txt", TY_state, fmt="%.6f")
    np.savetxt(f"{data_folder}/idx.txt", idx, fmt="%d")

    return minimum_error_pointwise, minimum_diff_pointwise, minimum_diff_pointwise_index, TY_state, idx