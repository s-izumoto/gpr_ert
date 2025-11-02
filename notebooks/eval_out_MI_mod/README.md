
---

## 🚀 Pipeline Workflow

1. **Generate resistivity fields**  
   ```bash
   python scripts/01_make_fields.py --config configs/data/make_fields.yml
   ```

2. **Fit PCA and project fields**  
   ```bash
   python scripts/02_fit_pca_and_project.py --config configs/pca/pca_randomized.yml
   ```

3. **Simulate ERT measurements (pyGIMLi physics forward)**  
   ```bash
   python scripts/03_make_surrogate_pairs_pygimli.py --config configs/simulate/make_surrogate_pairs.yml
   ```
   ```bash
   python scripts/03_make_surrogate_pairs_wenner.py --config configs/simulate/make_surrogate_pairs_wenner.yml
   ```

---
