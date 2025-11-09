
---

# GPR–ERT Sequential Design and Evaluation Pipeline

This repository implements a full workflow for **synthetic Electrical Resistivity Tomography (ERT)** and **Gaussian Process Regression (GPR)**–based **sequential experimental design**.  
It covers all stages from generating heterogeneous resistivity fields to inversion and quantitative evaluation against a Wenner array–based baseline.
> ℹ️ For an overview of key terms such as ERT, GPR, inverse analysis, and electrode array, please refer to the “📘 Background” section at the end of this README.

# GPR-Based Sequential Design: Performance Summary

**Compared to the array–based baseline**, the GPR-based sequential design — using a **Mutual Information–based acquisition function** — shows consistent performance gains in overall metrics. Example comparison images can be found in the `images_example/` folder.
- **IoU after Otsu thresholding and morphological filtering** improves by approximately **8 %**,  
- **RMSE** by **6 %**,  
- **Pearson r** and **Spearman ρ** by **4 %**, and  
- **Jensen–Shannon divergence (JSD)** by **2 %** on average.  
- Notably, in the **deeper half of the domain**, the improvements are more pronounced, with **Pearson r** increasing by **18 %**, **Spearman ρ** by **16 %**, and **RMSE** decreasing by **9 %**.
- In the **shallow half**, all metrics also show moderate improvements of a few percent, and **no degradation was observed** in any of the evaluation criteria.

## 🛠️ Installation

This repository provides an `environment.yml` file to reproduce the full software environment.

### Quick start (recommended)
```bash
# 0) (Optional) Install Mambaforge or Miniconda
#    https://conda-forge.org/miniforge/

# 1) Create the environment (public name, no user-specific paths)
mamba env create -n gpr-ert -f environment.yml    # if you have mamba
# OR
conda env create -n gpr-ert -f environment.yml

# 2) Activate it
conda activate gpr-ert

# 3) Verify
python -V
pip -V
```

---

## 🧭 Project Structure

The repository is organized into clearly separated layers for configuration, execution, and core functionality.  
This modular structure allows the pipeline to be easily reproduced, debugged, or extended.

```
gpr_ert/
├── build/                       # Core library of reusable modules and core logic
│   ├── __init__.py
│   ├── make_fields.py           # Generate synthetic 2D resistivity maps
│   ├── fit_pca.py               # Train and apply PCA to resistivity maps
│   ├── cluster_pca.py           # Cluster PCA-projected fields and select representative samples
│   ├── design.py                # ERT design-space enumerator (ABMN generation, metric features)
│   ├── measurements_warmup.py   # Forward simulation for initial (warm-up) Wenner measurements
│   ├── measurements_post_warmup.py  # Forward simulations for extended array types
│   ├── gpr_seq_core.py          # Core Gaussian Process Regression (GPR) implementation
│   ├── sequential_GPR.py        # Sequential design driver wrapping sequential_GPR
│   ├── invert_GPR.py            # Batch inversion of GPR-selected measurements
│   ├── forward_invert_Wenner.py # Forward and inversion workflow for Wenner array-based baseline
│   └── evaluate_GPR_vs_Wenner.py# Comparison metrics and evaluation functions
│
├── scripts/                     # Thin CLI wrappers calling corresponding build modules
│   ├── 01_make_fields.py
│   ├── 02_fit_pca.py
│   ├── 03_cluster_pca.py
│   ├── 04_measurements_warmup.py
│   ├── 05_measurements_post_warmup.py
│   ├── 06_sequential_GPR.py
│   ├── 07_invert_GPR.py
│   ├── 08_forward_invert_Wenner.py
│   ├── 09_evaluate_GPR_vs_Wenner.py
│   ├── 10_plot_summary.py
|   └── XX_make_images.py
│
├── configs/                     # YAML configuration files for reproducible experiments
│   ├── make_fields.yml          # Parameters for synthetic field generation (domain size, randomness)
│   ├── fit_pca.yml              # PCA configuration (n_components, explained variance)
│   ├── cluster_pca.yml          # Clustering parameters (n_clusters, algorithm, sampling mode)
│   ├── measurements_warmup.yml  # Wenner array warm-up forward modeling parameters
│   ├── measurements_post_warmup.yml # Multi-array forward modeling setup
│   ├── sequential_GPR.yml       # GPR sequential design setup (kernel, acquisition, warmup length)
│   ├── invert_GPR.yml           # Inversion setup for sequential GPR outputs
│   ├── forward_invert_Wenner.yml# Wenner array-based baseline forward/inverse modeling setup
│   └── evaluate_GPR_vs_Wenner.yml # Evaluation and metric comparison settings
│
├── README.md                    # Main documentation (project overview and workflow)
└── environment.yml              # Python dependencies for reproducibility
```

### Explanation

- **build/** — contains the scientific and computational core.  
  Each module can be imported and tested independently; all scripts rely on these internal APIs.  
- **scripts/** — minimal command-line wrappers that load YAML configs and call the appropriate function in `build/`.  
- **configs/** — contains all experiment parameters. Changing YAML files allows fully reproducible reruns.  

This structure ensures **clear separation between logic, execution, and configuration**, enabling transparent and reproducible research workflows.


## Overview of the Pipeline

The workflow is composed of ten modular scripts executed **in numerical order (01–10)**.  
Each step consumes the previous output and produces the next stage’s input.

| Step | Script | Purpose |
|------|---------|----------|
| **01** | `01_make_fields.py` | Generate synthetic 2-D resistivity fields representing a wide range of geological scenarios. |
| **02** | `02_fit_pca.py` | Train a PCA model to compress and reconstruct resistivity fields into a low-dimensional latent space. |
| **03** | `03_cluster_pca.py` | Cluster latent vectors to identify representative samples (centroids/medoids). |
| **04** | `04_measurements_warmup.py` | Simulate ERT measurements (Wenner-alpha) to collect the **warm-up dataset** used to initialize GPR. |
| **05** | `05_measurements_post_warmup.py` | Extend the dataset with candidate measurements from multiple array types (Schlumberger, Dipole-Dipole, Gradient, etc.). |
| **06** | `06_sequential_GPR.py` | Apply **Gaussian Process Regression (GPR)** after the warm-up period. The GPR is first fitted on the warm-up data, then iteratively selects new measurements using acquisition functions (UCB, LCB, EI, MI, MAXVAR). |
| **07** | `07_invert_GPR.py` | Invert the data obtained through the sequential GPR process to reconstruct the subsurface resistivity distribution. |
| **08** | `08_forward_invert_Wenner.py` | Perform forward modeling and inversion using the Wenner array-based baseline configuration for reference. |
| **09** | `09_evaluate_GPR_vs_Wenner.py` | Quantitatively evaluate the inversion results from GPR-based and Wenner array-based measurements using error and spatial similarity metrics. |
| **10** | `10_plot_summary.py` | Summarize and visualize evaluation metrics (boxplots, relative improvements, subset analysis). |

---

## The Warm-up Phase and Sequential GPR

Before the active GPR phase begins, a **warm-up period** is used to collect initial data and stabilize the surrogate model.  

- During this period, a **fixed set of measurement configurations** (typically Wenner-alpha arrays) is simulated.  
- The resulting data form the **warm-up dataset**, which provides the first few known input–output pairs for the Gaussian Process.  
- The **sequential GPR phase** then starts: at each step, the GPR model estimates uncertainty over the remaining candidate measurements, applies an **acquisition function** (e.g., Mutual Information or Expected Improvement), and selects the next configuration to sample.  
- The newly observed data point is added to the training set, and the GPR is re-fitted — progressively improving the surrogate model.

This strategy balances **exploration** (reducing model uncertainty) and **exploitation** (targeting high-impact regions), while maintaining continuity between the warm-up and active phases.

---

## Conceptual Flow

1. **Synthetic field generation:** Create heterogeneous resistivity fields that mimic realistic geological heterogeneity.  
2. **Dimensionality reduction:** Use PCA to map resistivity maps to a compact latent space.  
3. **Sample selection:** Choose representative cases for simulation.  
4. **Warm-up forward modeling:** Collect initial ERT responses using a standard array configuration.  
5. **Candidate expansion:** Simulate responses for a large set of potential array configurations.  
6. **Sequential GPR:** Start from the warm-up GP fit, apply acquisition functions to iteratively select the next best measurement configurations.  
7. **Inversion:** Reconstruct the resistivity field based on sequentially acquired data.  
8. **Baseline inversion:** Perform equivalent inversion for the Wenner reference.  
9. **Evaluation:** Compare the reconstructed and true fields using statistical and spatial similarity metrics.  
10. **Visualization:** Produce comparative summaries and visual figures.

---

## Key Features

- **Warm-up → Active GPR workflow:** separates data collection and adaptive learning phases.  
- **Multiple array families:** Wenner, Schlumberger, Dipole–Dipole, Gradient.  
- **Acquisition functions:** UCB, LCB, EI, MAXVAR, and Mutual Information.  
- **Separable kernels:** distinct RBFs for dipole distance and position, with automatic length-scale tracking.  
- **YAML-driven configuration** for reproducible experiments.  
- **Per-field logs** (CSV, NPZ) recording kernel hyperparameters, acquisition statistics, and inversion quality.

---

## Typical Outputs

- Synthetic and reconstructed resistivity maps  
- PCA components and latent representations  
- Warm-up and post-warm-up measurement datasets  
- Sequential GPR logs and kernel diagnostics  
- Inversion models (GPR vs Wenner)  
- Evaluation summaries and visual plots

---
## ⚙️ End-to-End Pipeline Workflow

This section summarizes the **full reproducible workflow** implemented in this repository.  
Each step is executed in order, using its own YAML configuration file under `configs/`.  
The workflow spans from **synthetic field generation** to **inversion, evaluation, and visualization**.

---

1. **Generate synthetic resistivity fields**  
   Create diverse 2-D subsurface resistivity maps representing different geological conditions.  
   ```bash
   python scripts/01_make_fields.py --config configs/make_fields.yml
   ```

2. **Fit PCA and project fields into latent space**  
   Train a PCA model to compress the resistivity maps and obtain their latent representations.  
   ```bash
   python scripts/02_fit_pca.py --config configs/fit_pca.yml
   ```

3. **Cluster latent representations**  
   Group the PCA-projected fields into clusters and select representative samples (centroids or medoids).  
   ```bash
   python scripts/03_cluster_pca.py --config configs/cluster_pca.yml
   ```

4. **Warm-up measurement simulation (Wenner array)**  
   Run forward ERT simulations using pyGIMLi with a **Wenner-alpha array** to collect the initial *warm-up* dataset.  
   These data are used to initialize the Gaussian Process surrogate model.  
   ```bash
   python scripts/04_measurements_warmup.py --config configs/measurements_warmup.yml
   ```

5. **Extended measurement simulation (multiple arrays)**  
   Simulate ERT responses for additional array types (Schlumberger, Dipole-Dipole, Gradient, etc.) to build the candidate pool for active design.  
   ```bash
   python scripts/05_measurements_post_warmup.py --config configs/measurements_post_warmup.yml
   ```

6. **Sequential Gaussian Process Regression (GPR)**  
   Fit the GPR model on the warm-up data, then enter the **active learning phase**:  
   at each step, use an acquisition function (UCB, LCB, EI, MI, or MAXVAR) to select the next most informative measurement.  
   ```bash
   python scripts/06_sequential_GPR.py --config configs/sequential_GPR.yml
   ```

7. **Inversion of sequential GPR results**  
   Perform inversion using only the measurements selected by the sequential GPR process to reconstruct the subsurface resistivity model.  
   ```bash
   python scripts/07_invert_GPR.py --config configs/invert_GPR.yml
   ```

8. **Forward and inverse modeling with the Wenner array-based baseline**  
   Run the same modeling and inversion pipeline using the Wenner array to establish a reference baseline.  
   ```bash
   python scripts/08_forward_invert_Wenner.py --config configs/forward_invert_Wenner.yml
   ```

9. **Quantitative evaluation of GPR vs. Wenner results**  
   Compare inversion results from the GPR-based sequential design and the Wenner array-based baseline using statistical and spatial similarity metrics.  
   ```bash
   python scripts/09_evaluate_GPR_vs_Wenner.py --config configs/evaluate_GPR_vs_Wenner.yml
   ```

10. **Generate summary plots and reports**  
    Visualize the evaluation results — boxplots, relative improvements, and field-wise summaries — for final analysis.  
    ```bash
    python scripts/10_plot_summary.py --config configs/plot_summary.yml
    ```

---

## 📘 Background

### Electrical Resistivity Tomography (ERT)
**Electrical Resistivity Tomography (ERT)** is a **geophysical imaging technique** used to infer subsurface structure and moisture distribution from **electrical resistivity**.  
Multiple electrodes are installed on the ground surface or in boreholes. **A pair of electrodes (A, B)** injects current, while **another pair (M, N)** measures the resulting potential difference.  
By repeating this process with many electrode combinations, one can obtain information to estimate the **subsurface resistivity distribution (ρ)**.  
Because resistivity depends on **water content, salinity, and soil or rock type**, ERT enables non-destructive investigation of processes such as **infiltration, saltwater intrusion, contaminant transport, and subsurface structure characterization**.

---

### What Are Electrode Arrays?
An **electrode array** is a **measurement design scheme** defining how multiple electrodes (e.g., 32) are placed on the surface and how electrode pairs (A–B for current, M–N for potential) are combined to perform repeated measurements.  
When 32 electrodes are used, the number of possible (A–B–M–N) combinations is enormous, and the order and spacing of these combinations significantly affect measurement depth, sensitivity, and noise characteristics.  
This **systematic enumeration of electrode combinations** is referred to as an *electrode array*. Representative types include:

| Array Type | Characteristics | Typical Use |
|:--|:--|:--|
| **Wenner Array** | Four electrodes (A–M–N–B) equally spaced; simple combination generation | Robust to noise; suitable for shallow investigations |
| **Schlumberger Array** | Current electrodes (A, B) placed far apart; potential electrodes (M, N) close together | High sensitivity to deeper structures |
| **Dipole–Dipole Array** | Current and potential pairs separated | High spatial resolution; more sensitive to noise |
| **Gradient Array** | Fixed current electrodes with multiple potential electrodes measured simultaneously | High measurement efficiency and dense data coverage |

In this repository, the **Wenner–alpha array** is adopted as the reference configuration because of its symmetrical sensitivity distribution and suitability for comparison with **GPR-based sequential design**.

---

### What Is Inversion?
ERT measurements provide **potential differences at the surface** corresponding to each electrode combination,  
but what we truly seek is the **subsurface resistivity distribution** itself.  
Thus, we must numerically determine a subsurface model that best explains the observed data—this process is known as **inversion**.  
Here, inversion is performed using the open-source geophysical library **pyGIMLi**, yielding **reconstructed subsurface resistivity distributions** based on sequentially selected measurement series.  
These inversion results are later used to **evaluate the performance of the GPR-based sequential design**.

---

### pyGIMLi (Python Geophysical Inversion and Modelling Library)
**pyGIMLi** is an open-source library for numerical simulation and inversion in geophysical exploration.  
It supports ERT, Induced Polarization (IP), and Spectral Induced Polarization (SIP), and performs **forward modelling, sensitivity analysis, and inversion** using the finite-element method (FEM).  
In this repository, pyGIMLi is employed to **simulate both forward and inverse ERT computations using the Wenner array**,  
allowing synthetic **apparent resistivity maps** to be generated from input conductivity fields.  
These simulated “true” datasets are then used as **inputs and benchmarks** for evaluating the GPR sequential design.

---

### Sequential Design via Gaussian Process Regression (GPR)
**Gaussian Process Regression (GPR)** models the **statistical correlation** between observation points.  
For ERT-type measurements, GPR can infer **unmeasured electrode combinations (A–B–M–N)** from previously observed results and determine **which combination would provide the greatest information gain next**.  
In this repository, GPR-based sequential design is applied not to the inverted subsurface model,  
but to the **measurement data space (potential differences)** before inversion.  
GPR learns correlations among electrode combinations and **sequentially generates measurement series that maximize efficiency**.  
The effectiveness of this design is then evaluated by performing inversion on the GPR-selected measurement series and comparing the **reconstructed subsurface resistivity distributions** against those from the reference Wenner configuration.

---

### 🧩 Summary
- **Electrical Resistivity Tomography (ERT)** estimates subsurface resistivity distributions by injecting current and measuring potential differences across many electrode combinations.  
- An **electrode array** defines how multiple electrodes (e.g., 32) are combined to perform these measurements, controlling sensitivity and depth characteristics.  
- **Inversion** reconstructs the subsurface resistivity distribution from measured data and is essential for evaluating the outcomes of GPR sequential design.  
- **pyGIMLi** provides the numerical foundation for forward and inverse modelling in this workflow.  
- The **GPR-based sequential design** operates in the *pre-inversion measurement data space*, and its effectiveness is quantitatively assessed through **inversion-based reconstruction accuracy** relative to the Wenner baseline.
