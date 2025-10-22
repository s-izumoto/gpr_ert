
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

4. **Build oracle masks (high-gradient regions)**  
   ```bash
   python scripts/04_make_oracle_masks.py --config configs/oracle/make_oracle.yml
   ```

5. **Reduce & diversify oracle masks (indices only)**  
   ```bash
   python scripts/04b_reduce_oracle_diversity.py --config configs/oracle/reduce_oracle.yml
   ```

6. **Train surrogate model (ERT approximation)**  
   ```bash
   python scripts/05_train_surrogate.py --config configs/surrogate/train_surrogate.yml
   ```

7. **Evaluate surrogate model**  
   ```bash
   python scripts/06_eval_surrogate.py --config configs/eval/eval_surrogate.yml
   ```

---
