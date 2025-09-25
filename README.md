# StochasticPaSR

A Python package for stochastic modeling of the **Partially Stirred Reactor (PaSR)** using the **Monte Carlo particle method**.

---

## Purpose

This project was developed to **generate synthetic datasets** intended for feeding and supporting different computational and machine learning models. In particular, it has been utilized in the Partially Stirred Reactor (PaSR) model to implement *on-the-fly* **adaptive chemical time scale formulation**. By providing a flexible and scalable data generation framework, the project facilitates more accurate modeling of complex chemically reacting flows by incorporating both detailed reaction kinetics and mixing dynamics, thereby reducing reliance on costly high-fidelity datasets.

---

## Functionalities

The code consists of two main components:

1. **Data Generation**  
   - Run simulations via `StochasticModelPaSR.py`  
   - Core simulation functions are in `StochasticFunctions.py`  
   - Reaction rates can be computed using **two alternative approaches**:  
     - **LFR (Laminar Finite Rate)** method  
     - **ODE (Ordinary Differential Equation) solver**  
   - Includes three different **mixing models**:  
     - **IEM (Interaction by Exchange with the Mean)** [1]
     - **MC (Modified Curl)** [2]
     - **KerM (Kernel Constrained)** [3]

2. **Post-Processing**  
   - `PaSR_post.py` gathers and analyzes generated data  
   - Performs error estimation and comparison between the LFR- and ODE-based reaction rate calculations  
   - Selects the **most optimal chemical time scale formulation** based on these evaluations  

> ⚠️ Designed for **Python 3.10+**. All dependencies are listed in `pyproject.toml`.

---

## Requirements

- Python 3.10  
- numpy ≥ 1.24  
- pandas ≥ 2.0  
- matplotlib ≥ 3.7  
- scipy ≥ 1.11  
- cantera ≥ 3.0  
- pyyaml ≥ 6.0  
- PyCSP ([GitHub link](https://github.com/rmalpica/PyCSP))  

> ⚠️ PyCSP currently only works on Python 3.10.

---

## Installation

### 1. Docker Installation

1. Install Docker for your OS: [Docker Installation Guide](https://docs.docker.com/engine/install/)  
2. Download the code  
3. Build the Docker image:

```bash
docker build -t stochasticpasr .
docker run --rm -it stochasticpasr
```

### 2. Conda Installation and Test
```bash
# 1. Clone the repository
git clone https://github.com/yalcinkayay/StochasticPaSR.git
cd StochasticPaSR

# 2. Create a Conda environment with Python 3.10
conda create -n stochasticpasr python=3.10
conda activate stochasticpasr

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyCSP from GitHub
pip install git+https://github.com/rmalpica/PyCSP.git

# 5. Install project from pyproject.toml
pip install .

# 6. Test
python StochasticModelPaSR.py

```
## References  

[1] C. Dopazo, *Relaxation of initial probability density functions in the turbulent convection of scalar fields*, Physics of Fluids, 22(1), pp. 20–30, 1979. https://doi.org/10.1063/1.862431  

[2] R. L. Curl, *Dispersed phase mixing: I. Theory and effects in simple reactors*, AIChE Journal, 9(2), pp. 175–181, 1963. https://doi.org/10.1002/aic.690090207  

[3] X. Su, J. Wei, X. Wang, H. Zhou, E. R. Hawkes, Z. Ren, *A pairwise mixing model with kernel constraint and its appraisal in transported PDF simulations of ethylene flames*, Combustion and Flame, 255, 112916, 2023. https://doi.org/10.1016/j.combustflame.2023.112916  
