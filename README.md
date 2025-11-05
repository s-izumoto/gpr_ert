
---

# GPR–ERT Sequential Design and Evaluation Pipeline

This repository implements a full workflow for **synthetic Electrical Resistivity Tomography (ERT)** and **Gaussian Process Regression (GPR)**–based **sequential experimental design**.  
It covers all stages from generating heterogeneous resistivity fields to inversion and quantitative evaluation against a Wenner baseline.

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
│   ├── sequential_GPR.py        # Sequential design driver wrapping gpr_seq_core
│   ├── invert_GPR.py            # Batch inversion of GPR-selected measurements
│   ├── forward_invert_Wenner.py # Forward and inversion workflow for Wenner baseline
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
│   └── 10_plot_summary.py
│
├── configs/                     # YAML configuration files for reproducible experiments
│   ├── make_fields.yml          # Parameters for synthetic field generation (domain size, randomness)
│   ├── fit_pca.yml              # PCA configuration (n_components, explained variance)
│   ├── cluster_pca.yml          # Clustering parameters (n_clusters, algorithm, sampling mode)
│   ├── measurements_warmup.yml  # Wenner warm-up forward modeling parameters
│   ├── measurements_post_warmup.yml # Multi-array forward modeling setup
│   ├── sequential_GPR.yml       # GPR sequential design setup (kernel, acquisition, warmup length)
│   ├── invert_GPR.yml           # Inversion setup for sequential GPR outputs
│   ├── forward_invert_Wenner.yml# Wenner baseline forward/inverse modeling setup
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
| **08** | `08_forward_invert_Wenner.py` | Perform forward modeling and inversion using the Wenner baseline configuration for reference. |
| **09** | `09_evaluate_GPR_vs_Wenner.py` | Quantitatively evaluate the inversion results from GPR-based and Wenner-based measurements using error and spatial similarity metrics. |
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

8. **Forward and inverse modeling with the Wenner baseline**  
   Run the same modeling and inversion pipeline using the Wenner configuration to establish a reference baseline.  
   ```bash
   python scripts/08_forward_invert_Wenner.py --config configs/forward_invert_Wenner.yml
   ```

9. **Quantitative evaluation of GPR vs. Wenner results**  
   Compare inversion results from the GPR-based sequential design and the Wenner baseline using statistical and spatial similarity metrics.  
   ```bash
   python scripts/09_evaluate_GPR_vs_Wenner.py --config configs/evaluate_GPR_vs_Wenner.yml
   ```

10. **Generate summary plots and reports**  
    Visualize the evaluation results — boxplots, relative improvements, and field-wise summaries — for final analysis.  
    ```bash
    python scripts/10_plot_summary.py --config configs/plot_summary.yml
    ```

---
