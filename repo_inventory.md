# Repository Inventory

```
C:\Users\izumoto\github\go-boed-ert
|-- .gitignore
|-- .vscode\
|   `-- settings.json
|-- README.md
|-- configs\
|   |-- data\
|   |   `-- make_fields.yml
|   |-- eval\
|   |   `-- eval_surrogate.yml
|   |-- oracle\
|   |   |-- make_oracle.yml
|   |   `-- reduce_oracle.yml
|   |-- pca\
|   |   `-- pca_randomized.yml
|   |-- simulate\
|   |   |-- make_surrogate_pairs.yml
|   |   `-- make_surrogate_pairs_wenner.yml
|   `-- surrogate\
|       `-- train_surrogate.yml
|-- data\
|   |-- .gitkeep
|   |-- eval_outputs\
|   |   `-- field0\
|   |       |-- outliers_field0.csv
|   |       |-- outliers_field10.csv
|   |       |-- outliers_field100.csv
|   |       |-- outliers_field1000.csv
|   |       |-- outliers_field2000.csv
|   |       |-- parity_(field_0).png
|   |       |-- parity_(field_10).png
|   |       |-- parity_(field_100).png
|   |       |-- parity_(field_1000).png
|   |       |-- parity_(field_2000).png
|   |       |-- pred_vs_ratio_(field_0).png
|   |       |-- pred_vs_ratio_(field_10).png
|   |       |-- pred_vs_ratio_(field_100).png
|   |       |-- pred_vs_ratio_(field_1000).png
|   |       |-- pred_vs_ratio_(field_2000).png
|   |       |-- reciprocity_delta_hist.png
|   |       |-- residual_hist_(field_0).png
|   |       |-- residual_hist_(field_10).png
|   |       |-- residual_hist_(field_100).png
|   |       |-- residual_hist_(field_1000).png
|   |       |-- residual_hist_(field_2000).png
|   |       `-- summary.json
|   |-- indexes\
|   |   `-- .gitkeep
|   |-- interim\
|   |   |-- .gitkeep
|   |   |-- ert_wenner_subset\
|   |   |   |-- ert_surrogate_wenner.npz
|   |   |   |-- ert_surrogate_wenner_reduced.npz
|   |   |   `-- field_source_idx.npy
|   |   |-- fields\
|   |   |   |-- dataset.npz
|   |   |   |-- label_map.json
|   |   |   |-- meta.json
|   |   |   `-- previews\
|   |   |       |-- preview_GEOLOGY_1.png
|   |   |       |-- preview_GEOLOGY_2.png
|   |   |       |-- preview_GEOLOGY_3.png
|   |   |       |-- preview_GEOLOGY_4.png
|   |   |       |-- preview_RESISTIVE_1.png
|   |   |       |-- preview_RESISTIVE_2.png
|   |   |       |-- preview_RESISTIVE_3.png
|   |   |       |-- preview_RESISTIVE_4.png
|   |   |       |-- preview_SEAWATER_1.png
|   |   |       |-- preview_SEAWATER_2.png
|   |   |       |-- preview_SEAWATER_3.png
|   |   |       |-- preview_SEAWATER_4.png
|   |   |       |-- preview_SURFACE_1.png
|   |   |       |-- preview_SURFACE_2.png
|   |   |       |-- preview_SURFACE_3.png
|   |   |       |-- preview_SURFACE_4.png
|   |   |       |-- preview_TRACER_1.png
|   |   |       |-- preview_TRACER_2.png
|   |   |       |-- preview_TRACER_3.png
|   |   |       |-- preview_TRACER_4.png
|   |   |       |-- preview_WATERTABLE_1.png
|   |   |       |-- preview_WATERTABLE_2.png
|   |   |       |-- preview_WATERTABLE_3.png
|   |   |       `-- preview_WATERTABLE_4.png
|   |   |-- pca\
|   |   |   |-- Z.npz
|   |   |   |-- pca_joint.joblib
|   |   |   |-- pca_joint_meta.json
|   |   |   `-- previews_joint_all\
|   |   |       |-- joint_all_01.png
|   |   |       |-- joint_all_02.png
|   |   |       |-- joint_all_03.png
|   |   |       |-- joint_all_04.png
|   |   |       |-- joint_all_05.png
|   |   |       `-- joint_all_06.png
|   |   `-- surrogate_ds\
|   |       |-- ert_surrogate.npz
|   |       |-- ert_surrogate_reduced.npz
|   |       `-- field_source_idx.npy
|   `-- processed\
|       |-- .gitkeep
|       |-- oracle_pairs\
|       |   |-- latent_oracle_pairs_aligned.npz
|       |   |-- oracle_preview_roi_field0.png
|       |   |-- oracle_preview_roi_field1.png
|       |   |-- oracle_preview_roi_field2.png
|       |   |-- oracle_preview_roi_field3.png
|       |   |-- oracle_preview_roi_field4.png
|       |   |-- oracle_preview_roi_field5.png
|       |   |-- oracle_preview_roi_field6.png
|       |   |-- oracle_preview_roi_field7.png
|       |   |-- oracle_preview_roi_field8.png
|       |   `-- oracle_preview_roi_field9.png
|       `-- oracle_pairs_reduced\
|           |-- oracle_pairs_reduced_dedup.npz
|           |-- oracle_pairs_reduced_dedup_indices.npy
|           |-- oracle_pairs_reduced_diverse.npz
|           |-- oracle_pairs_reduced_diverse_indices.npy
|           |-- oracle_pairs_reduced_diverse_indices.npz
|           `-- oracle_pairs_reduced_reduction_README.txt
|-- environment.yml
|-- figures\
|   `-- .gitkeep
|-- logs\
|   `-- .gitkeep
|-- models\
|   `-- ert_surrogate\
|       |-- best_model.pt
|       |-- best_model_with_residual.sd.pt
|       |-- best_model_with_residual.ts.pt
|       |-- ckpt.pt
|       |-- metrics.json
|       `-- scaler_meta.npz
|-- notebooks\
|   |-- 05_filter_surrogate_by_reduction.py
|   |   def _rows_per_field_from_meta(npz)
|   |   def _extract_Z_fields_any(M, rows_per_field)
|   |   def _row_hashes(X)
|   |   def _load_npz_or_npy(path)
|   |   def _extract_Z_from_mapping(M, prefer_keys)
|   |   def _extract_indices_or_mask(M)
|   |   def _infer_sample_length(npz)
|   |   def _candidate_sample_keys(npz, N)
|   |   def _keep_idx_by_Z(Z_red, Z_all, verbose)
|   |   def _match_reduced_to_surrogate(reduced_any, surrogate_npz, force_z, source_Z_any, verbose, rows_per_field)
|   |   def _filter_npz_to_out(in_path, keep_fields_idx, out_path, rows_per_field, verbose)
|   |   def main()
|   |-- dataset_ert.py
|   |   def load_ert_npz(ert_npz_path)
|   |   def collate_fn(batch)
|   |       class HParams:
|   |   def __init__(self)
|   |       class ERTOracleDataset(Dataset):
|   |   def __init__(self, ert_npz_path, oracle_npz_path, hp, pick_scale_index, max_fields, field_offset, warmup_npz_path, warmup_pick, active_select, active_seed, patterns_per_field)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |-- environment.yml
|   |-- jacob_global.py
|   |   def kl_beta_torch(a1, b1, a0, b0, eps)
|   |   def normalize_np(v, eps)
|   |   def make_fixed_warmup_designs(K)
|   |   def make_true_field_and_labels()
|   |   def _get_surrogate()
|   |   def sample_designs_and_jacobians(M, N, rng, field_img)
|   |   def measure_scalar(design4, true_g)
|   |   def eig_weights_from_p(p, sigma)
|   |   def make_teacher_deltaE(J_row, p, tau, lam, sigma, eps)
|   |       class SurrogateJacobian:
|   |   def __init__(self, model_dir, pca_joblib, device)
|   |   def _geom_logk_from_Dn(self, Dn_raw)
|   |   def _featurize_designs_raw(self, Dn_raw)
|   |   def _field_to_flat_crop(self, field_img)
|   |   def _project_to_Zs(self, flat_crop, require_grad)
|   |   def jacobians_wrt_pixels(self, designs4, field_img, batch)
|   |       class DeepSetsEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden, out_dim)
|   |   def forward(self, S)
|   |       class FiLMDecoder(nn.Module):
|   |   def __init__(self, global_dim, design_dim, pix_dim, mlp_hidden)
|   |   def forward(self, h_global, design)
|   |       class HistHead(nn.Module):
|   |   def __init__(self, d_model, n_pix)
|   |   def forward(self, h)
|   |       class EviNet(nn.Module):
|   |   def __init__(self, meas_in_dim)
|   |   def forward_hist(self, S_set)
|   |   def forward_next(self, S_set, design)
|   |   def forward(self, S_set, design)
|   |-- jacob_global1.py
|   |   def kl_beta_torch(a1, b1, a0, b0, eps)
|   |   def normalize_np(v, eps)
|   |   def make_fixed_warmup_designs(K)
|   |   def make_true_field_and_labels()
|   |   def sample_designs_and_jacobians(M, N, rng)
|   |   def measure_scalar(design4, true_g)
|   |   def eig_weights_from_p(p, sigma)
|   |   def make_teacher_deltaE(J_row, p, tau, lam, sigma, eps)
|   |       class DeepSetsEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden, out_dim)
|   |   def forward(self, S)
|   |       class FiLMDecoder(nn.Module):
|   |   def __init__(self, global_dim, design_dim, pix_dim, mlp_hidden)
|   |   def forward(self, h_global, design)
|   |       class HistHead(nn.Module):
|   |   def __init__(self, d_model, n_pix)
|   |   def forward(self, h)
|   |       class EviNet(nn.Module):
|   |   def __init__(self, meas_in_dim)
|   |   def forward_hist(self, S_set)
|   |   def forward_next(self, S_set, design)
|   |   def forward(self, S_set, design)
|   |-- outputs\
|   |   |-- toy_model.pt
|   |   `-- toy_training_log.txt
|   |-- outputs_evi\
|   |   |-- evi_ckpt_20251024_230512.pt
|   |   `-- evi_ckpt_20251024_231033.pt
|   |-- outputs_test_single\
|   |   |-- field_0000\
|   |   |   |-- p_step_033.npy
|   |   |   |-- p_step_033.png
|   |   |   |-- p_step_034.npy
|   |   |   |-- p_step_034.png
|   |   |   |-- p_step_035.npy
|   |   |   |-- p_step_035.png
|   |   |   |-- p_step_036.npy
|   |   |   |-- p_step_036.png
|   |   |   |-- p_step_037.npy
|   |   |   |-- p_step_037.png
|   |   |   |-- p_step_038.npy
|   |   |   |-- p_step_038.png
|   |   |   |-- p_step_039.npy
|   |   |   |-- p_step_039.png
|   |   |   |-- p_step_040.npy
|   |   |   |-- p_step_040.png
|   |   |   |-- p_step_041.npy
|   |   |   |-- p_step_041.png
|   |   |   |-- p_step_042.npy
|   |   |   |-- p_step_042.png
|   |   |   |-- p_step_043.npy
|   |   |   |-- p_step_043.png
|   |   |   |-- p_step_044.npy
|   |   |   |-- p_step_044.png
|   |   |   |-- p_step_045.npy
|   |   |   |-- p_step_045.png
|   |   |   |-- p_step_046.npy
|   |   |   |-- p_step_046.png
|   |   |   |-- p_step_047.npy
|   |   |   |-- p_step_047.png
|   |   |   |-- p_step_048.npy
|   |   |   |-- p_step_048.png
|   |   |   |-- p_step_049.npy
|   |   |   |-- p_step_049.png
|   |   |   |-- p_step_050.npy
|   |   |   |-- p_step_050.png
|   |   |   |-- p_step_051.npy
|   |   |   |-- p_step_051.png
|   |   |   |-- p_step_052.npy
|   |   |   |-- p_step_052.png
|   |   |   |-- p_step_053.npy
|   |   |   |-- p_step_053.png
|   |   |   |-- p_step_054.npy
|   |   |   |-- p_step_054.png
|   |   |   |-- p_step_055.npy
|   |   |   |-- p_step_055.png
|   |   |   |-- p_step_056.npy
|   |   |   |-- p_step_056.png
|   |   |   |-- p_step_057.npy
|   |   |   |-- p_step_057.png
|   |   |   |-- p_step_058.npy
|   |   |   |-- p_step_058.png
|   |   |   |-- p_step_059.npy
|   |   |   |-- p_step_059.png
|   |   |   |-- p_step_060.npy
|   |   |   |-- p_step_060.png
|   |   |   |-- p_step_061.npy
|   |   |   |-- p_step_061.png
|   |   |   |-- p_step_062.npy
|   |   |   |-- p_step_062.png
|   |   |   |-- p_step_063.npy
|   |   |   |-- p_step_063.png
|   |   |   |-- p_step_064.npy
|   |   |   |-- p_step_064.png
|   |   |   |-- p_step_065.npy
|   |   |   |-- p_step_065.png
|   |   |   |-- p_step_066.npy
|   |   |   |-- p_step_066.png
|   |   |   |-- p_step_067.npy
|   |   |   |-- p_step_067.png
|   |   |   |-- p_step_068.npy
|   |   |   |-- p_step_068.png
|   |   |   |-- p_step_069.npy
|   |   |   |-- p_step_069.png
|   |   |   |-- p_step_070.npy
|   |   |   |-- p_step_070.png
|   |   |   |-- p_step_071.npy
|   |   |   |-- p_step_071.png
|   |   |   |-- p_step_072.npy
|   |   |   |-- p_step_072.png
|   |   |   |-- p_step_073.npy
|   |   |   |-- p_step_073.png
|   |   |   |-- p_step_074.npy
|   |   |   |-- p_step_074.png
|   |   |   |-- p_step_075.npy
|   |   |   |-- p_step_075.png
|   |   |   |-- p_step_076.npy
|   |   |   |-- p_step_076.png
|   |   |   |-- p_step_077.npy
|   |   |   |-- p_step_077.png
|   |   |   |-- p_step_078.npy
|   |   |   |-- p_step_078.png
|   |   |   |-- p_step_079.npy
|   |   |   |-- p_step_079.png
|   |   |   |-- p_step_080.npy
|   |   |   |-- p_step_080.png
|   |   |   |-- p_step_081.npy
|   |   |   |-- p_step_081.png
|   |   |   |-- p_step_082.npy
|   |   |   |-- p_step_082.png
|   |   |   |-- p_step_083.npy
|   |   |   |-- p_step_083.png
|   |   |   |-- p_step_084.npy
|   |   |   |-- p_step_084.png
|   |   |   |-- p_step_085.npy
|   |   |   |-- p_step_085.png
|   |   |   |-- p_step_086.npy
|   |   |   |-- p_step_086.png
|   |   |   |-- p_step_087.npy
|   |   |   |-- p_step_087.png
|   |   |   |-- p_step_088.npy
|   |   |   |-- p_step_088.png
|   |   |   |-- p_step_089.npy
|   |   |   |-- p_step_089.png
|   |   |   |-- p_step_090.npy
|   |   |   |-- p_step_090.png
|   |   |   |-- p_step_091.npy
|   |   |   |-- p_step_091.png
|   |   |   |-- p_step_092.npy
|   |   |   |-- p_step_092.png
|   |   |   |-- p_step_093.npy
|   |   |   |-- p_step_093.png
|   |   |   |-- p_step_094.npy
|   |   |   |-- p_step_094.png
|   |   |   |-- p_step_095.npy
|   |   |   |-- p_step_095.png
|   |   |   |-- p_step_096.npy
|   |   |   |-- p_step_096.png
|   |   |   |-- p_step_097.npy
|   |   |   |-- p_step_097.png
|   |   |   |-- p_step_098.npy
|   |   |   |-- p_step_098.png
|   |   |   |-- p_step_099.npy
|   |   |   |-- p_step_099.png
|   |   |   |-- p_step_100.npy
|   |   |   |-- p_step_100.png
|   |   |   |-- p_step_101.npy
|   |   |   |-- p_step_101.png
|   |   |   |-- p_step_102.npy
|   |   |   |-- p_step_102.png
|   |   |   |-- p_step_103.npy
|   |   |   |-- p_step_103.png
|   |   |   |-- p_step_104.npy
|   |   |   |-- p_step_104.png
|   |   |   |-- p_step_105.npy
|   |   |   |-- p_step_105.png
|   |   |   |-- p_step_106.npy
|   |   |   |-- p_step_106.png
|   |   |   |-- p_step_107.npy
|   |   |   |-- p_step_107.png
|   |   |   |-- p_step_108.npy
|   |   |   |-- p_step_108.png
|   |   |   |-- p_step_109.npy
|   |   |   |-- p_step_109.png
|   |   |   |-- p_step_110.npy
|   |   |   |-- p_step_110.png
|   |   |   |-- p_step_111.npy
|   |   |   |-- p_step_111.png
|   |   |   |-- p_step_112.npy
|   |   |   |-- p_step_112.png
|   |   |   |-- p_step_113.npy
|   |   |   |-- p_step_113.png
|   |   |   |-- p_step_114.npy
|   |   |   |-- p_step_114.png
|   |   |   |-- p_step_115.npy
|   |   |   |-- p_step_115.png
|   |   |   |-- p_step_116.npy
|   |   |   |-- p_step_116.png
|   |   |   |-- p_step_117.npy
|   |   |   |-- p_step_117.png
|   |   |   |-- p_step_118.npy
|   |   |   |-- p_step_118.png
|   |   |   |-- p_step_119.npy
|   |   |   |-- p_step_119.png
|   |   |   |-- p_step_120.npy
|   |   |   |-- p_step_120.png
|   |   |   |-- p_step_121.npy
|   |   |   |-- p_step_121.png
|   |   |   |-- p_step_122.npy
|   |   |   |-- p_step_122.png
|   |   |   |-- p_step_123.npy
|   |   |   |-- p_step_123.png
|   |   |   |-- p_step_124.npy
|   |   |   |-- p_step_124.png
|   |   |   |-- p_step_125.npy
|   |   |   |-- p_step_125.png
|   |   |   |-- p_step_126.npy
|   |   |   |-- p_step_126.png
|   |   |   |-- p_step_127.npy
|   |   |   |-- p_step_127.png
|   |   |   |-- p_step_128.npy
|   |   |   |-- p_step_128.png
|   |   |   |-- p_step_129.npy
|   |   |   |-- p_step_129.png
|   |   |   |-- p_step_130.npy
|   |   |   |-- p_step_130.png
|   |   |   |-- p_step_131.npy
|   |   |   |-- p_step_131.png
|   |   |   |-- p_step_132.npy
|   |   |   |-- p_step_132.png
|   |   |   |-- p_step_133.npy
|   |   |   |-- p_step_133.png
|   |   |   |-- p_step_134.npy
|   |   |   |-- p_step_134.png
|   |   |   |-- p_step_135.npy
|   |   |   |-- p_step_135.png
|   |   |   |-- p_step_136.npy
|   |   |   |-- p_step_136.png
|   |   |   |-- p_step_137.npy
|   |   |   |-- p_step_137.png
|   |   |   |-- p_step_138.npy
|   |   |   |-- p_step_138.png
|   |   |   |-- p_step_139.npy
|   |   |   |-- p_step_139.png
|   |   |   |-- p_step_140.npy
|   |   |   |-- p_step_140.png
|   |   |   |-- p_step_141.npy
|   |   |   |-- p_step_141.png
|   |   |   |-- p_step_142.npy
|   |   |   |-- p_step_142.png
|   |   |   |-- p_step_143.npy
|   |   |   |-- p_step_143.png
|   |   |   |-- p_step_144.npy
|   |   |   |-- p_step_144.png
|   |   |   |-- p_step_145.npy
|   |   |   |-- p_step_145.png
|   |   |   |-- p_step_146.npy
|   |   |   |-- p_step_146.png
|   |   |   |-- p_step_147.npy
|   |   |   |-- p_step_147.png
|   |   |   |-- p_step_148.npy
|   |   |   |-- p_step_148.png
|   |   |   |-- p_step_149.npy
|   |   |   |-- p_step_149.png
|   |   |   |-- p_step_150.npy
|   |   |   |-- p_step_150.png
|   |   |   |-- p_step_151.npy
|   |   |   |-- p_step_151.png
|   |   |   |-- p_step_152.npy
|   |   |   |-- p_step_152.png
|   |   |   |-- p_step_153.npy
|   |   |   |-- p_step_153.png
|   |   |   |-- p_step_154.npy
|   |   |   |-- p_step_154.png
|   |   |   |-- p_step_155.npy
|   |   |   |-- p_step_155.png
|   |   |   |-- pdiff_step_037.npy
|   |   |   |-- pdiff_step_037.png
|   |   |   |-- pdiff_step_038.npy
|   |   |   |-- pdiff_step_038.png
|   |   |   |-- pdiff_step_039.npy
|   |   |   |-- pdiff_step_039.png
|   |   |   |-- pdiff_step_040.npy
|   |   |   |-- pdiff_step_040.png
|   |   |   |-- pdiff_step_041.npy
|   |   |   |-- pdiff_step_041.png
|   |   |   |-- pdiff_step_042.npy
|   |   |   |-- pdiff_step_042.png
|   |   |   |-- pdiff_step_043.npy
|   |   |   |-- pdiff_step_043.png
|   |   |   |-- pdiff_step_044.npy
|   |   |   |-- pdiff_step_044.png
|   |   |   |-- pdiff_step_045.npy
|   |   |   |-- pdiff_step_045.png
|   |   |   |-- pdiff_step_046.npy
|   |   |   |-- pdiff_step_046.png
|   |   |   |-- pdiff_step_047.npy
|   |   |   |-- pdiff_step_047.png
|   |   |   |-- pdiff_step_048.npy
|   |   |   |-- pdiff_step_048.png
|   |   |   |-- pdiff_step_049.npy
|   |   |   |-- pdiff_step_049.png
|   |   |   |-- pdiff_step_050.npy
|   |   |   |-- pdiff_step_050.png
|   |   |   |-- pdiff_step_051.npy
|   |   |   |-- pdiff_step_051.png
|   |   |   |-- pdiff_step_052.npy
|   |   |   |-- pdiff_step_052.png
|   |   |   |-- pdiff_step_053.npy
|   |   |   |-- pdiff_step_053.png
|   |   |   |-- pdiff_step_054.npy
|   |   |   |-- pdiff_step_054.png
|   |   |   |-- pdiff_step_055.npy
|   |   |   |-- pdiff_step_055.png
|   |   |   |-- pdiff_step_056.npy
|   |   |   |-- pdiff_step_056.png
|   |   |   |-- pdiff_step_057.npy
|   |   |   |-- pdiff_step_057.png
|   |   |   |-- pdiff_step_058.npy
|   |   |   |-- pdiff_step_058.png
|   |   |   |-- pdiff_step_059.npy
|   |   |   |-- pdiff_step_059.png
|   |   |   |-- pdiff_step_060.npy
|   |   |   |-- pdiff_step_060.png
|   |   |   |-- pdiff_step_061.npy
|   |   |   |-- pdiff_step_061.png
|   |   |   |-- pdiff_step_062.npy
|   |   |   |-- pdiff_step_062.png
|   |   |   |-- pdiff_step_063.npy
|   |   |   |-- pdiff_step_063.png
|   |   |   |-- pdiff_step_064.npy
|   |   |   |-- pdiff_step_064.png
|   |   |   |-- pdiff_step_065.npy
|   |   |   |-- pdiff_step_065.png
|   |   |   |-- pdiff_step_066.npy
|   |   |   |-- pdiff_step_066.png
|   |   |   |-- pdiff_step_067.npy
|   |   |   |-- pdiff_step_067.png
|   |   |   |-- pdiff_step_068.npy
|   |   |   |-- pdiff_step_068.png
|   |   |   |-- pdiff_step_069.npy
|   |   |   |-- pdiff_step_069.png
|   |   |   |-- pdiff_step_070.npy
|   |   |   |-- pdiff_step_070.png
|   |   |   |-- pdiff_step_071.npy
|   |   |   |-- pdiff_step_071.png
|   |   |   |-- pdiff_step_072.npy
|   |   |   |-- pdiff_step_072.png
|   |   |   |-- pdiff_step_073.npy
|   |   |   |-- pdiff_step_073.png
|   |   |   |-- pdiff_step_074.npy
|   |   |   |-- pdiff_step_074.png
|   |   |   |-- pdiff_step_075.npy
|   |   |   |-- pdiff_step_075.png
|   |   |   |-- pdiff_step_076.npy
|   |   |   |-- pdiff_step_076.png
|   |   |   |-- pdiff_step_077.npy
|   |   |   |-- pdiff_step_077.png
|   |   |   |-- pdiff_step_078.npy
|   |   |   |-- pdiff_step_078.png
|   |   |   |-- pdiff_step_079.npy
|   |   |   |-- pdiff_step_079.png
|   |   |   |-- pdiff_step_080.npy
|   |   |   |-- pdiff_step_080.png
|   |   |   |-- pdiff_step_081.npy
|   |   |   |-- pdiff_step_081.png
|   |   |   |-- pdiff_step_082.npy
|   |   |   |-- pdiff_step_082.png
|   |   |   |-- pdiff_step_083.npy
|   |   |   |-- pdiff_step_083.png
|   |   |   |-- pdiff_step_084.npy
|   |   |   |-- pdiff_step_084.png
|   |   |   |-- pdiff_step_085.npy
|   |   |   |-- pdiff_step_085.png
|   |   |   |-- pdiff_step_086.npy
|   |   |   |-- pdiff_step_086.png
|   |   |   |-- pdiff_step_087.npy
|   |   |   |-- pdiff_step_087.png
|   |   |   |-- pdiff_step_088.npy
|   |   |   |-- pdiff_step_088.png
|   |   |   |-- pdiff_step_089.npy
|   |   |   |-- pdiff_step_089.png
|   |   |   |-- pdiff_step_090.npy
|   |   |   |-- pdiff_step_090.png
|   |   |   |-- pdiff_step_091.npy
|   |   |   |-- pdiff_step_091.png
|   |   |   |-- pdiff_step_092.npy
|   |   |   |-- pdiff_step_092.png
|   |   |   |-- pdiff_step_093.npy
|   |   |   |-- pdiff_step_093.png
|   |   |   |-- pdiff_step_094.npy
|   |   |   |-- pdiff_step_094.png
|   |   |   |-- pdiff_step_095.npy
|   |   |   |-- pdiff_step_095.png
|   |   |   |-- pdiff_step_096.npy
|   |   |   |-- pdiff_step_096.png
|   |   |   |-- pdiff_step_097.npy
|   |   |   |-- pdiff_step_097.png
|   |   |   |-- pdiff_step_098.npy
|   |   |   |-- pdiff_step_098.png
|   |   |   |-- pdiff_step_099.npy
|   |   |   |-- pdiff_step_099.png
|   |   |   |-- pdiff_step_100.npy
|   |   |   |-- pdiff_step_100.png
|   |   |   |-- pdiff_step_101.npy
|   |   |   |-- pdiff_step_101.png
|   |   |   |-- pdiff_step_102.npy
|   |   |   |-- pdiff_step_102.png
|   |   |   |-- pdiff_step_103.npy
|   |   |   |-- pdiff_step_103.png
|   |   |   |-- pdiff_step_104.npy
|   |   |   |-- pdiff_step_104.png
|   |   |   |-- pdiff_step_105.npy
|   |   |   |-- pdiff_step_105.png
|   |   |   |-- pdiff_step_106.npy
|   |   |   |-- pdiff_step_106.png
|   |   |   |-- pdiff_step_107.npy
|   |   |   |-- pdiff_step_107.png
|   |   |   |-- pdiff_step_108.npy
|   |   |   |-- pdiff_step_108.png
|   |   |   |-- pdiff_step_109.npy
|   |   |   |-- pdiff_step_109.png
|   |   |   |-- pdiff_step_110.npy
|   |   |   |-- pdiff_step_110.png
|   |   |   |-- pdiff_step_111.npy
|   |   |   |-- pdiff_step_111.png
|   |   |   |-- pdiff_step_112.npy
|   |   |   |-- pdiff_step_112.png
|   |   |   |-- pdiff_step_113.npy
|   |   |   |-- pdiff_step_113.png
|   |   |   |-- pdiff_step_114.npy
|   |   |   |-- pdiff_step_114.png
|   |   |   |-- pdiff_step_115.npy
|   |   |   |-- pdiff_step_115.png
|   |   |   |-- pdiff_step_116.npy
|   |   |   |-- pdiff_step_116.png
|   |   |   |-- pdiff_step_117.npy
|   |   |   |-- pdiff_step_117.png
|   |   |   |-- pdiff_step_118.npy
|   |   |   |-- pdiff_step_118.png
|   |   |   |-- pdiff_step_119.npy
|   |   |   |-- pdiff_step_119.png
|   |   |   |-- pdiff_step_120.npy
|   |   |   |-- pdiff_step_120.png
|   |   |   |-- pdiff_step_121.npy
|   |   |   |-- pdiff_step_121.png
|   |   |   |-- pdiff_step_122.npy
|   |   |   |-- pdiff_step_122.png
|   |   |   |-- pdiff_step_123.npy
|   |   |   |-- pdiff_step_123.png
|   |   |   |-- pdiff_step_124.npy
|   |   |   |-- pdiff_step_124.png
|   |   |   |-- pdiff_step_125.npy
|   |   |   |-- pdiff_step_125.png
|   |   |   |-- pdiff_step_126.npy
|   |   |   |-- pdiff_step_126.png
|   |   |   |-- pdiff_step_127.npy
|   |   |   |-- pdiff_step_127.png
|   |   |   |-- pdiff_step_128.npy
|   |   |   |-- pdiff_step_128.png
|   |   |   |-- pdiff_step_129.npy
|   |   |   |-- pdiff_step_129.png
|   |   |   |-- pdiff_step_130.npy
|   |   |   |-- pdiff_step_130.png
|   |   |   |-- pdiff_step_131.npy
|   |   |   |-- pdiff_step_131.png
|   |   |   |-- pdiff_step_132.npy
|   |   |   |-- pdiff_step_132.png
|   |   |   |-- pdiff_step_133.npy
|   |   |   |-- pdiff_step_133.png
|   |   |   |-- pdiff_step_134.npy
|   |   |   |-- pdiff_step_134.png
|   |   |   |-- pdiff_step_135.npy
|   |   |   |-- pdiff_step_135.png
|   |   |   |-- pdiff_step_136.npy
|   |   |   |-- pdiff_step_136.png
|   |   |   |-- pdiff_step_137.npy
|   |   |   |-- pdiff_step_137.png
|   |   |   |-- pdiff_step_138.npy
|   |   |   |-- pdiff_step_138.png
|   |   |   |-- pdiff_step_139.npy
|   |   |   |-- pdiff_step_139.png
|   |   |   |-- pdiff_step_140.npy
|   |   |   |-- pdiff_step_140.png
|   |   |   |-- pdiff_step_141.npy
|   |   |   |-- pdiff_step_141.png
|   |   |   |-- pdiff_step_142.npy
|   |   |   |-- pdiff_step_142.png
|   |   |   |-- pdiff_step_143.npy
|   |   |   |-- pdiff_step_143.png
|   |   |   |-- pdiff_step_144.npy
|   |   |   |-- pdiff_step_144.png
|   |   |   |-- pdiff_step_145.npy
|   |   |   |-- pdiff_step_145.png
|   |   |   |-- pdiff_step_146.npy
|   |   |   |-- pdiff_step_146.png
|   |   |   |-- pdiff_step_147.npy
|   |   |   |-- pdiff_step_147.png
|   |   |   |-- pdiff_step_148.npy
|   |   |   |-- pdiff_step_148.png
|   |   |   |-- pdiff_step_149.npy
|   |   |   |-- pdiff_step_149.png
|   |   |   |-- pdiff_step_150.npy
|   |   |   |-- pdiff_step_150.png
|   |   |   |-- pdiff_step_151.npy
|   |   |   |-- pdiff_step_151.png
|   |   |   |-- pdiff_step_152.npy
|   |   |   |-- pdiff_step_152.png
|   |   |   |-- pdiff_step_153.npy
|   |   |   |-- pdiff_step_153.png
|   |   |   |-- pdiff_step_154.npy
|   |   |   |-- pdiff_step_154.png
|   |   |   |-- pdiff_step_155.npy
|   |   |   |-- pdiff_step_155.png
|   |   |   |-- pinst_diff_step_037.npy
|   |   |   |-- pinst_diff_step_037.png
|   |   |   |-- pinst_diff_step_038.npy
|   |   |   |-- pinst_diff_step_038.png
|   |   |   |-- pinst_diff_step_039.npy
|   |   |   |-- pinst_diff_step_039.png
|   |   |   |-- pinst_diff_step_040.npy
|   |   |   |-- pinst_diff_step_040.png
|   |   |   |-- pinst_diff_step_041.npy
|   |   |   |-- pinst_diff_step_041.png
|   |   |   |-- pinst_diff_step_042.npy
|   |   |   |-- pinst_diff_step_042.png
|   |   |   |-- pinst_diff_step_043.npy
|   |   |   |-- pinst_diff_step_043.png
|   |   |   |-- pinst_diff_step_044.npy
|   |   |   |-- pinst_diff_step_044.png
|   |   |   |-- pinst_diff_step_045.npy
|   |   |   |-- pinst_diff_step_045.png
|   |   |   |-- pinst_diff_step_046.npy
|   |   |   |-- pinst_diff_step_046.png
|   |   |   |-- pinst_diff_step_047.npy
|   |   |   |-- pinst_diff_step_047.png
|   |   |   |-- pinst_diff_step_048.npy
|   |   |   |-- pinst_diff_step_048.png
|   |   |   |-- pinst_diff_step_049.npy
|   |   |   |-- pinst_diff_step_049.png
|   |   |   |-- pinst_diff_step_050.npy
|   |   |   |-- pinst_diff_step_050.png
|   |   |   |-- pinst_diff_step_051.npy
|   |   |   |-- pinst_diff_step_051.png
|   |   |   |-- pinst_diff_step_052.npy
|   |   |   |-- pinst_diff_step_052.png
|   |   |   |-- pinst_diff_step_053.npy
|   |   |   |-- pinst_diff_step_053.png
|   |   |   |-- pinst_diff_step_054.npy
|   |   |   |-- pinst_diff_step_054.png
|   |   |   |-- pinst_diff_step_055.npy
|   |   |   |-- pinst_diff_step_055.png
|   |   |   |-- pinst_diff_step_056.npy
|   |   |   |-- pinst_diff_step_056.png
|   |   |   |-- pinst_diff_step_057.npy
|   |   |   |-- pinst_diff_step_057.png
|   |   |   |-- pinst_diff_step_058.npy
|   |   |   |-- pinst_diff_step_058.png
|   |   |   |-- pinst_diff_step_059.npy
|   |   |   |-- pinst_diff_step_059.png
|   |   |   |-- pinst_diff_step_060.npy
|   |   |   |-- pinst_diff_step_060.png
|   |   |   |-- pinst_diff_step_061.npy
|   |   |   |-- pinst_diff_step_061.png
|   |   |   |-- pinst_diff_step_062.npy
|   |   |   |-- pinst_diff_step_062.png
|   |   |   |-- pinst_diff_step_063.npy
|   |   |   |-- pinst_diff_step_063.png
|   |   |   |-- pinst_diff_step_064.npy
|   |   |   |-- pinst_diff_step_064.png
|   |   |   |-- pinst_diff_step_065.npy
|   |   |   |-- pinst_diff_step_065.png
|   |   |   |-- pinst_diff_step_066.npy
|   |   |   |-- pinst_diff_step_066.png
|   |   |   |-- pinst_diff_step_067.npy
|   |   |   |-- pinst_diff_step_067.png
|   |   |   |-- pinst_diff_step_068.npy
|   |   |   |-- pinst_diff_step_068.png
|   |   |   |-- pinst_diff_step_069.npy
|   |   |   |-- pinst_diff_step_069.png
|   |   |   |-- pinst_diff_step_070.npy
|   |   |   |-- pinst_diff_step_070.png
|   |   |   |-- pinst_diff_step_071.npy
|   |   |   |-- pinst_diff_step_071.png
|   |   |   |-- pinst_diff_step_072.npy
|   |   |   |-- pinst_diff_step_072.png
|   |   |   |-- pinst_diff_step_073.npy
|   |   |   |-- pinst_diff_step_073.png
|   |   |   |-- pinst_diff_step_074.npy
|   |   |   |-- pinst_diff_step_074.png
|   |   |   |-- pinst_diff_step_075.npy
|   |   |   |-- pinst_diff_step_075.png
|   |   |   |-- pinst_diff_step_076.npy
|   |   |   |-- pinst_diff_step_076.png
|   |   |   |-- pinst_diff_step_077.npy
|   |   |   |-- pinst_diff_step_077.png
|   |   |   |-- pinst_diff_step_078.npy
|   |   |   |-- pinst_diff_step_078.png
|   |   |   |-- pinst_diff_step_079.npy
|   |   |   |-- pinst_diff_step_079.png
|   |   |   |-- pinst_diff_step_080.npy
|   |   |   |-- pinst_diff_step_080.png
|   |   |   |-- pinst_diff_step_081.npy
|   |   |   |-- pinst_diff_step_081.png
|   |   |   |-- pinst_diff_step_082.npy
|   |   |   |-- pinst_diff_step_082.png
|   |   |   |-- pinst_diff_step_083.npy
|   |   |   |-- pinst_diff_step_083.png
|   |   |   |-- pinst_diff_step_084.npy
|   |   |   |-- pinst_diff_step_084.png
|   |   |   |-- pinst_diff_step_085.npy
|   |   |   |-- pinst_diff_step_085.png
|   |   |   |-- pinst_diff_step_086.npy
|   |   |   |-- pinst_diff_step_086.png
|   |   |   |-- pinst_diff_step_087.npy
|   |   |   |-- pinst_diff_step_087.png
|   |   |   |-- pinst_diff_step_088.npy
|   |   |   |-- pinst_diff_step_088.png
|   |   |   |-- pinst_diff_step_089.npy
|   |   |   |-- pinst_diff_step_089.png
|   |   |   |-- pinst_diff_step_090.npy
|   |   |   |-- pinst_diff_step_090.png
|   |   |   |-- pinst_diff_step_091.npy
|   |   |   |-- pinst_diff_step_091.png
|   |   |   |-- pinst_diff_step_092.npy
|   |   |   |-- pinst_diff_step_092.png
|   |   |   |-- pinst_diff_step_093.npy
|   |   |   |-- pinst_diff_step_093.png
|   |   |   |-- pinst_diff_step_094.npy
|   |   |   |-- pinst_diff_step_094.png
|   |   |   |-- pinst_diff_step_095.npy
|   |   |   |-- pinst_diff_step_095.png
|   |   |   |-- pinst_diff_step_096.npy
|   |   |   |-- pinst_diff_step_096.png
|   |   |   |-- pinst_diff_step_097.npy
|   |   |   |-- pinst_diff_step_097.png
|   |   |   |-- pinst_diff_step_098.npy
|   |   |   |-- pinst_diff_step_098.png
|   |   |   |-- pinst_diff_step_099.npy
|   |   |   |-- pinst_diff_step_099.png
|   |   |   |-- pinst_diff_step_100.npy
|   |   |   |-- pinst_diff_step_100.png
|   |   |   |-- pinst_diff_step_101.npy
|   |   |   |-- pinst_diff_step_101.png
|   |   |   |-- pinst_diff_step_102.npy
|   |   |   |-- pinst_diff_step_102.png
|   |   |   |-- pinst_diff_step_103.npy
|   |   |   |-- pinst_diff_step_103.png
|   |   |   |-- pinst_diff_step_104.npy
|   |   |   |-- pinst_diff_step_104.png
|   |   |   |-- pinst_diff_step_105.npy
|   |   |   |-- pinst_diff_step_105.png
|   |   |   |-- pinst_diff_step_106.npy
|   |   |   |-- pinst_diff_step_106.png
|   |   |   |-- pinst_diff_step_107.npy
|   |   |   |-- pinst_diff_step_107.png
|   |   |   |-- pinst_diff_step_108.npy
|   |   |   |-- pinst_diff_step_108.png
|   |   |   |-- pinst_diff_step_109.npy
|   |   |   |-- pinst_diff_step_109.png
|   |   |   |-- pinst_diff_step_110.npy
|   |   |   |-- pinst_diff_step_110.png
|   |   |   |-- pinst_diff_step_111.npy
|   |   |   |-- pinst_diff_step_111.png
|   |   |   |-- pinst_diff_step_112.npy
|   |   |   |-- pinst_diff_step_112.png
|   |   |   |-- pinst_diff_step_113.npy
|   |   |   |-- pinst_diff_step_113.png
|   |   |   |-- pinst_diff_step_114.npy
|   |   |   |-- pinst_diff_step_114.png
|   |   |   |-- pinst_diff_step_115.npy
|   |   |   |-- pinst_diff_step_115.png
|   |   |   |-- pinst_diff_step_116.npy
|   |   |   |-- pinst_diff_step_116.png
|   |   |   |-- pinst_diff_step_117.npy
|   |   |   |-- pinst_diff_step_117.png
|   |   |   |-- pinst_diff_step_118.npy
|   |   |   |-- pinst_diff_step_118.png
|   |   |   |-- pinst_diff_step_119.npy
|   |   |   |-- pinst_diff_step_119.png
|   |   |   |-- pinst_diff_step_120.npy
|   |   |   |-- pinst_diff_step_120.png
|   |   |   |-- pinst_diff_step_121.npy
|   |   |   |-- pinst_diff_step_121.png
|   |   |   |-- pinst_diff_step_122.npy
|   |   |   |-- pinst_diff_step_122.png
|   |   |   |-- pinst_diff_step_123.npy
|   |   |   |-- pinst_diff_step_123.png
|   |   |   |-- pinst_diff_step_124.npy
|   |   |   |-- pinst_diff_step_124.png
|   |   |   |-- pinst_diff_step_125.npy
|   |   |   |-- pinst_diff_step_125.png
|   |   |   |-- pinst_diff_step_126.npy
|   |   |   |-- pinst_diff_step_126.png
|   |   |   |-- pinst_diff_step_127.npy
|   |   |   |-- pinst_diff_step_127.png
|   |   |   |-- pinst_diff_step_128.npy
|   |   |   |-- pinst_diff_step_128.png
|   |   |   |-- pinst_diff_step_129.npy
|   |   |   |-- pinst_diff_step_129.png
|   |   |   |-- pinst_diff_step_130.npy
|   |   |   |-- pinst_diff_step_130.png
|   |   |   |-- pinst_diff_step_131.npy
|   |   |   |-- pinst_diff_step_131.png
|   |   |   |-- pinst_diff_step_132.npy
|   |   |   |-- pinst_diff_step_132.png
|   |   |   |-- pinst_diff_step_133.npy
|   |   |   |-- pinst_diff_step_133.png
|   |   |   |-- pinst_diff_step_134.npy
|   |   |   |-- pinst_diff_step_134.png
|   |   |   |-- pinst_diff_step_135.npy
|   |   |   |-- pinst_diff_step_135.png
|   |   |   |-- pinst_diff_step_136.npy
|   |   |   |-- pinst_diff_step_136.png
|   |   |   |-- pinst_diff_step_137.npy
|   |   |   |-- pinst_diff_step_137.png
|   |   |   |-- pinst_diff_step_138.npy
|   |   |   |-- pinst_diff_step_138.png
|   |   |   |-- pinst_diff_step_139.npy
|   |   |   |-- pinst_diff_step_139.png
|   |   |   |-- pinst_diff_step_140.npy
|   |   |   |-- pinst_diff_step_140.png
|   |   |   |-- pinst_diff_step_141.npy
|   |   |   |-- pinst_diff_step_141.png
|   |   |   |-- pinst_diff_step_142.npy
|   |   |   |-- pinst_diff_step_142.png
|   |   |   |-- pinst_diff_step_143.npy
|   |   |   |-- pinst_diff_step_143.png
|   |   |   |-- pinst_diff_step_144.npy
|   |   |   |-- pinst_diff_step_144.png
|   |   |   |-- pinst_diff_step_145.npy
|   |   |   |-- pinst_diff_step_145.png
|   |   |   |-- pinst_diff_step_146.npy
|   |   |   |-- pinst_diff_step_146.png
|   |   |   |-- pinst_diff_step_147.npy
|   |   |   |-- pinst_diff_step_147.png
|   |   |   |-- pinst_diff_step_148.npy
|   |   |   |-- pinst_diff_step_148.png
|   |   |   |-- pinst_diff_step_149.npy
|   |   |   |-- pinst_diff_step_149.png
|   |   |   |-- pinst_diff_step_150.npy
|   |   |   |-- pinst_diff_step_150.png
|   |   |   |-- pinst_diff_step_151.npy
|   |   |   |-- pinst_diff_step_151.png
|   |   |   |-- pinst_diff_step_152.npy
|   |   |   |-- pinst_diff_step_152.png
|   |   |   |-- pinst_diff_step_153.npy
|   |   |   |-- pinst_diff_step_153.png
|   |   |   |-- pinst_diff_step_154.npy
|   |   |   |-- pinst_diff_step_154.png
|   |   |   |-- pinst_diff_step_155.npy
|   |   |   |-- pinst_diff_step_155.png
|   |   |   |-- pinst_step_036.npy
|   |   |   |-- pinst_step_036.png
|   |   |   |-- pinst_step_037.npy
|   |   |   |-- pinst_step_037.png
|   |   |   |-- pinst_step_038.npy
|   |   |   |-- pinst_step_038.png
|   |   |   |-- pinst_step_039.npy
|   |   |   |-- pinst_step_039.png
|   |   |   |-- pinst_step_040.npy
|   |   |   |-- pinst_step_040.png
|   |   |   |-- pinst_step_041.npy
|   |   |   |-- pinst_step_041.png
|   |   |   |-- pinst_step_042.npy
|   |   |   |-- pinst_step_042.png
|   |   |   |-- pinst_step_043.npy
|   |   |   |-- pinst_step_043.png
|   |   |   |-- pinst_step_044.npy
|   |   |   |-- pinst_step_044.png
|   |   |   |-- pinst_step_045.npy
|   |   |   |-- pinst_step_045.png
|   |   |   |-- pinst_step_046.npy
|   |   |   |-- pinst_step_046.png
|   |   |   |-- pinst_step_047.npy
|   |   |   |-- pinst_step_047.png
|   |   |   |-- pinst_step_048.npy
|   |   |   |-- pinst_step_048.png
|   |   |   |-- pinst_step_049.npy
|   |   |   |-- pinst_step_049.png
|   |   |   |-- pinst_step_050.npy
|   |   |   |-- pinst_step_050.png
|   |   |   |-- pinst_step_051.npy
|   |   |   |-- pinst_step_051.png
|   |   |   |-- pinst_step_052.npy
|   |   |   |-- pinst_step_052.png
|   |   |   |-- pinst_step_053.npy
|   |   |   |-- pinst_step_053.png
|   |   |   |-- pinst_step_054.npy
|   |   |   |-- pinst_step_054.png
|   |   |   |-- pinst_step_055.npy
|   |   |   |-- pinst_step_055.png
|   |   |   |-- pinst_step_056.npy
|   |   |   |-- pinst_step_056.png
|   |   |   |-- pinst_step_057.npy
|   |   |   |-- pinst_step_057.png
|   |   |   |-- pinst_step_058.npy
|   |   |   |-- pinst_step_058.png
|   |   |   |-- pinst_step_059.npy
|   |   |   |-- pinst_step_059.png
|   |   |   |-- pinst_step_060.npy
|   |   |   |-- pinst_step_060.png
|   |   |   |-- pinst_step_061.npy
|   |   |   |-- pinst_step_061.png
|   |   |   |-- pinst_step_062.npy
|   |   |   |-- pinst_step_062.png
|   |   |   |-- pinst_step_063.npy
|   |   |   |-- pinst_step_063.png
|   |   |   |-- pinst_step_064.npy
|   |   |   |-- pinst_step_064.png
|   |   |   |-- pinst_step_065.npy
|   |   |   |-- pinst_step_065.png
|   |   |   |-- pinst_step_066.npy
|   |   |   |-- pinst_step_066.png
|   |   |   |-- pinst_step_067.npy
|   |   |   |-- pinst_step_067.png
|   |   |   |-- pinst_step_068.npy
|   |   |   |-- pinst_step_068.png
|   |   |   |-- pinst_step_069.npy
|   |   |   |-- pinst_step_069.png
|   |   |   |-- pinst_step_070.npy
|   |   |   |-- pinst_step_070.png
|   |   |   |-- pinst_step_071.npy
|   |   |   |-- pinst_step_071.png
|   |   |   |-- pinst_step_072.npy
|   |   |   |-- pinst_step_072.png
|   |   |   |-- pinst_step_073.npy
|   |   |   |-- pinst_step_073.png
|   |   |   |-- pinst_step_074.npy
|   |   |   |-- pinst_step_074.png
|   |   |   |-- pinst_step_075.npy
|   |   |   |-- pinst_step_075.png
|   |   |   |-- pinst_step_076.npy
|   |   |   |-- pinst_step_076.png
|   |   |   |-- pinst_step_077.npy
|   |   |   |-- pinst_step_077.png
|   |   |   |-- pinst_step_078.npy
|   |   |   |-- pinst_step_078.png
|   |   |   |-- pinst_step_079.npy
|   |   |   |-- pinst_step_079.png
|   |   |   |-- pinst_step_080.npy
|   |   |   |-- pinst_step_080.png
|   |   |   |-- pinst_step_081.npy
|   |   |   |-- pinst_step_081.png
|   |   |   |-- pinst_step_082.npy
|   |   |   |-- pinst_step_082.png
|   |   |   |-- pinst_step_083.npy
|   |   |   |-- pinst_step_083.png
|   |   |   |-- pinst_step_084.npy
|   |   |   |-- pinst_step_084.png
|   |   |   |-- pinst_step_085.npy
|   |   |   |-- pinst_step_085.png
|   |   |   |-- pinst_step_086.npy
|   |   |   |-- pinst_step_086.png
|   |   |   |-- pinst_step_087.npy
|   |   |   |-- pinst_step_087.png
|   |   |   |-- pinst_step_088.npy
|   |   |   |-- pinst_step_088.png
|   |   |   |-- pinst_step_089.npy
|   |   |   |-- pinst_step_089.png
|   |   |   |-- pinst_step_090.npy
|   |   |   |-- pinst_step_090.png
|   |   |   |-- pinst_step_091.npy
|   |   |   |-- pinst_step_091.png
|   |   |   |-- pinst_step_092.npy
|   |   |   |-- pinst_step_092.png
|   |   |   |-- pinst_step_093.npy
|   |   |   |-- pinst_step_093.png
|   |   |   |-- pinst_step_094.npy
|   |   |   |-- pinst_step_094.png
|   |   |   |-- pinst_step_095.npy
|   |   |   |-- pinst_step_095.png
|   |   |   |-- pinst_step_096.npy
|   |   |   |-- pinst_step_096.png
|   |   |   |-- pinst_step_097.npy
|   |   |   |-- pinst_step_097.png
|   |   |   |-- pinst_step_098.npy
|   |   |   |-- pinst_step_098.png
|   |   |   |-- pinst_step_099.npy
|   |   |   |-- pinst_step_099.png
|   |   |   |-- pinst_step_100.npy
|   |   |   |-- pinst_step_100.png
|   |   |   |-- pinst_step_101.npy
|   |   |   |-- pinst_step_101.png
|   |   |   |-- pinst_step_102.npy
|   |   |   |-- pinst_step_102.png
|   |   |   |-- pinst_step_103.npy
|   |   |   |-- pinst_step_103.png
|   |   |   |-- pinst_step_104.npy
|   |   |   |-- pinst_step_104.png
|   |   |   |-- pinst_step_105.npy
|   |   |   |-- pinst_step_105.png
|   |   |   |-- pinst_step_106.npy
|   |   |   |-- pinst_step_106.png
|   |   |   |-- pinst_step_107.npy
|   |   |   |-- pinst_step_107.png
|   |   |   |-- pinst_step_108.npy
|   |   |   |-- pinst_step_108.png
|   |   |   |-- pinst_step_109.npy
|   |   |   |-- pinst_step_109.png
|   |   |   |-- pinst_step_110.npy
|   |   |   |-- pinst_step_110.png
|   |   |   |-- pinst_step_111.npy
|   |   |   |-- pinst_step_111.png
|   |   |   |-- pinst_step_112.npy
|   |   |   |-- pinst_step_112.png
|   |   |   |-- pinst_step_113.npy
|   |   |   |-- pinst_step_113.png
|   |   |   |-- pinst_step_114.npy
|   |   |   |-- pinst_step_114.png
|   |   |   |-- pinst_step_115.npy
|   |   |   |-- pinst_step_115.png
|   |   |   |-- pinst_step_116.npy
|   |   |   |-- pinst_step_116.png
|   |   |   |-- pinst_step_117.npy
|   |   |   |-- pinst_step_117.png
|   |   |   |-- pinst_step_118.npy
|   |   |   |-- pinst_step_118.png
|   |   |   |-- pinst_step_119.npy
|   |   |   |-- pinst_step_119.png
|   |   |   |-- pinst_step_120.npy
|   |   |   |-- pinst_step_120.png
|   |   |   |-- pinst_step_121.npy
|   |   |   |-- pinst_step_121.png
|   |   |   |-- pinst_step_122.npy
|   |   |   |-- pinst_step_122.png
|   |   |   |-- pinst_step_123.npy
|   |   |   |-- pinst_step_123.png
|   |   |   |-- pinst_step_124.npy
|   |   |   |-- pinst_step_124.png
|   |   |   |-- pinst_step_125.npy
|   |   |   |-- pinst_step_125.png
|   |   |   |-- pinst_step_126.npy
|   |   |   |-- pinst_step_126.png
|   |   |   |-- pinst_step_127.npy
|   |   |   |-- pinst_step_127.png
|   |   |   |-- pinst_step_128.npy
|   |   |   |-- pinst_step_128.png
|   |   |   |-- pinst_step_129.npy
|   |   |   |-- pinst_step_129.png
|   |   |   |-- pinst_step_130.npy
|   |   |   |-- pinst_step_130.png
|   |   |   |-- pinst_step_131.npy
|   |   |   |-- pinst_step_131.png
|   |   |   |-- pinst_step_132.npy
|   |   |   |-- pinst_step_132.png
|   |   |   |-- pinst_step_133.npy
|   |   |   |-- pinst_step_133.png
|   |   |   |-- pinst_step_134.npy
|   |   |   |-- pinst_step_134.png
|   |   |   |-- pinst_step_135.npy
|   |   |   |-- pinst_step_135.png
|   |   |   |-- pinst_step_136.npy
|   |   |   |-- pinst_step_136.png
|   |   |   |-- pinst_step_137.npy
|   |   |   |-- pinst_step_137.png
|   |   |   |-- pinst_step_138.npy
|   |   |   |-- pinst_step_138.png
|   |   |   |-- pinst_step_139.npy
|   |   |   |-- pinst_step_139.png
|   |   |   |-- pinst_step_140.npy
|   |   |   |-- pinst_step_140.png
|   |   |   |-- pinst_step_141.npy
|   |   |   |-- pinst_step_141.png
|   |   |   |-- pinst_step_142.npy
|   |   |   |-- pinst_step_142.png
|   |   |   |-- pinst_step_143.npy
|   |   |   |-- pinst_step_143.png
|   |   |   |-- pinst_step_144.npy
|   |   |   |-- pinst_step_144.png
|   |   |   |-- pinst_step_145.npy
|   |   |   |-- pinst_step_145.png
|   |   |   |-- pinst_step_146.npy
|   |   |   |-- pinst_step_146.png
|   |   |   |-- pinst_step_147.npy
|   |   |   |-- pinst_step_147.png
|   |   |   |-- pinst_step_148.npy
|   |   |   |-- pinst_step_148.png
|   |   |   |-- pinst_step_149.npy
|   |   |   |-- pinst_step_149.png
|   |   |   |-- pinst_step_150.npy
|   |   |   |-- pinst_step_150.png
|   |   |   |-- pinst_step_151.npy
|   |   |   |-- pinst_step_151.png
|   |   |   |-- pinst_step_152.npy
|   |   |   |-- pinst_step_152.png
|   |   |   |-- pinst_step_153.npy
|   |   |   |-- pinst_step_153.png
|   |   |   |-- pinst_step_154.npy
|   |   |   |-- pinst_step_154.png
|   |   |   |-- pinst_step_155.npy
|   |   |   |-- pinst_step_155.png
|   |   |   `-- probs_steps_033_155.npz
|   |   `-- run_20251021_231932\
|   |       `-- field_0000\
|   |           |-- p_step_033.npy
|   |           |-- p_step_033.png
|   |           |-- p_step_034.npy
|   |           |-- p_step_034.png
|   |           |-- p_step_035.npy
|   |           |-- p_step_035.png
|   |           |-- p_step_036.npy
|   |           |-- p_step_036.png
|   |           |-- p_step_037.npy
|   |           |-- p_step_037.png
|   |           |-- p_step_038.npy
|   |           |-- p_step_038.png
|   |           |-- p_step_039.npy
|   |           |-- p_step_039.png
|   |           |-- p_step_040.npy
|   |           |-- p_step_040.png
|   |           |-- p_step_041.npy
|   |           |-- p_step_041.png
|   |           |-- p_step_042.npy
|   |           |-- p_step_042.png
|   |           |-- p_step_043.npy
|   |           |-- p_step_043.png
|   |           |-- p_step_044.npy
|   |           |-- p_step_044.png
|   |           |-- p_step_045.npy
|   |           |-- p_step_045.png
|   |           |-- p_step_046.npy
|   |           |-- p_step_046.png
|   |           |-- p_step_047.npy
|   |           |-- p_step_047.png
|   |           |-- p_step_048.npy
|   |           |-- p_step_048.png
|   |           |-- p_step_049.npy
|   |           |-- p_step_049.png
|   |           |-- p_step_050.npy
|   |           |-- p_step_050.png
|   |           |-- p_step_051.npy
|   |           |-- p_step_051.png
|   |           |-- p_step_052.npy
|   |           |-- p_step_052.png
|   |           |-- p_step_053.npy
|   |           |-- p_step_053.png
|   |           |-- p_step_054.npy
|   |           |-- p_step_054.png
|   |           |-- p_step_055.npy
|   |           |-- p_step_055.png
|   |           |-- p_step_056.npy
|   |           |-- p_step_056.png
|   |           |-- p_step_057.npy
|   |           |-- p_step_057.png
|   |           |-- p_step_058.npy
|   |           |-- p_step_058.png
|   |           |-- p_step_059.npy
|   |           |-- p_step_059.png
|   |           |-- p_step_060.npy
|   |           |-- p_step_060.png
|   |           |-- p_step_061.npy
|   |           |-- p_step_061.png
|   |           |-- p_step_062.npy
|   |           |-- p_step_062.png
|   |           |-- p_step_063.npy
|   |           |-- p_step_063.png
|   |           |-- p_step_064.npy
|   |           |-- p_step_064.png
|   |           |-- p_step_065.npy
|   |           |-- p_step_065.png
|   |           |-- p_step_066.npy
|   |           |-- p_step_066.png
|   |           |-- p_step_067.npy
|   |           |-- p_step_067.png
|   |           |-- p_step_068.npy
|   |           |-- p_step_068.png
|   |           |-- p_step_069.npy
|   |           |-- p_step_069.png
|   |           |-- p_step_070.npy
|   |           |-- p_step_070.png
|   |           |-- p_step_071.npy
|   |           |-- p_step_071.png
|   |           |-- p_step_072.npy
|   |           |-- p_step_072.png
|   |           |-- p_step_073.npy
|   |           |-- p_step_073.png
|   |           |-- p_step_074.npy
|   |           |-- p_step_074.png
|   |           |-- p_step_075.npy
|   |           |-- p_step_075.png
|   |           |-- p_step_076.npy
|   |           |-- p_step_076.png
|   |           |-- p_step_077.npy
|   |           |-- p_step_077.png
|   |           |-- p_step_078.npy
|   |           |-- p_step_078.png
|   |           |-- p_step_079.npy
|   |           |-- p_step_079.png
|   |           |-- p_step_080.npy
|   |           |-- p_step_080.png
|   |           |-- p_step_081.npy
|   |           |-- p_step_081.png
|   |           |-- p_step_082.npy
|   |           |-- p_step_082.png
|   |           |-- p_step_083.npy
|   |           |-- p_step_083.png
|   |           |-- p_step_084.npy
|   |           |-- p_step_084.png
|   |           |-- p_step_085.npy
|   |           |-- p_step_085.png
|   |           |-- p_step_086.npy
|   |           |-- p_step_086.png
|   |           |-- p_step_087.npy
|   |           |-- p_step_087.png
|   |           |-- p_step_088.npy
|   |           |-- p_step_088.png
|   |           |-- p_step_089.npy
|   |           |-- p_step_089.png
|   |           |-- p_step_090.npy
|   |           |-- p_step_090.png
|   |           |-- p_step_091.npy
|   |           |-- p_step_091.png
|   |           |-- p_step_092.npy
|   |           |-- p_step_092.png
|   |           |-- p_step_093.npy
|   |           |-- p_step_093.png
|   |           |-- p_step_094.npy
|   |           |-- p_step_094.png
|   |           |-- p_step_095.npy
|   |           |-- p_step_095.png
|   |           |-- p_step_096.npy
|   |           |-- p_step_096.png
|   |           |-- p_step_097.npy
|   |           |-- p_step_097.png
|   |           |-- p_step_098.npy
|   |           |-- p_step_098.png
|   |           |-- p_step_099.npy
|   |           |-- p_step_099.png
|   |           |-- p_step_100.npy
|   |           |-- p_step_100.png
|   |           |-- p_step_101.npy
|   |           |-- p_step_101.png
|   |           |-- p_step_102.npy
|   |           |-- p_step_102.png
|   |           |-- p_step_103.npy
|   |           |-- p_step_103.png
|   |           |-- p_step_104.npy
|   |           |-- p_step_104.png
|   |           |-- p_step_105.npy
|   |           |-- p_step_105.png
|   |           |-- p_step_106.npy
|   |           |-- p_step_106.png
|   |           |-- p_step_107.npy
|   |           |-- p_step_107.png
|   |           |-- p_step_108.npy
|   |           |-- p_step_108.png
|   |           |-- p_step_109.npy
|   |           |-- p_step_109.png
|   |           |-- p_step_110.npy
|   |           |-- p_step_110.png
|   |           |-- p_step_111.npy
|   |           |-- p_step_111.png
|   |           |-- p_step_112.npy
|   |           |-- p_step_112.png
|   |           |-- p_step_113.npy
|   |           |-- p_step_113.png
|   |           |-- p_step_114.npy
|   |           |-- p_step_114.png
|   |           |-- p_step_115.npy
|   |           |-- p_step_115.png
|   |           |-- p_step_116.npy
|   |           |-- p_step_116.png
|   |           |-- p_step_117.npy
|   |           |-- p_step_117.png
|   |           |-- p_step_118.npy
|   |           |-- p_step_118.png
|   |           |-- p_step_119.npy
|   |           |-- p_step_119.png
|   |           |-- p_step_120.npy
|   |           |-- p_step_120.png
|   |           |-- p_step_121.npy
|   |           |-- p_step_121.png
|   |           |-- p_step_122.npy
|   |           |-- p_step_122.png
|   |           |-- p_step_123.npy
|   |           |-- p_step_123.png
|   |           |-- p_step_124.npy
|   |           |-- p_step_124.png
|   |           |-- p_step_125.npy
|   |           |-- p_step_125.png
|   |           |-- p_step_126.npy
|   |           |-- p_step_126.png
|   |           |-- p_step_127.npy
|   |           |-- p_step_127.png
|   |           |-- p_step_128.npy
|   |           |-- p_step_128.png
|   |           |-- p_step_129.npy
|   |           |-- p_step_129.png
|   |           |-- p_step_130.npy
|   |           |-- p_step_130.png
|   |           |-- p_step_131.npy
|   |           |-- p_step_131.png
|   |           |-- p_step_132.npy
|   |           |-- p_step_132.png
|   |           |-- p_step_133.npy
|   |           |-- p_step_133.png
|   |           |-- p_step_134.npy
|   |           |-- p_step_134.png
|   |           |-- p_step_135.npy
|   |           |-- p_step_135.png
|   |           |-- p_step_136.npy
|   |           |-- p_step_136.png
|   |           |-- p_step_137.npy
|   |           |-- p_step_137.png
|   |           |-- p_step_138.npy
|   |           |-- p_step_138.png
|   |           |-- p_step_139.npy
|   |           |-- p_step_139.png
|   |           |-- p_step_140.npy
|   |           |-- p_step_140.png
|   |           |-- p_step_141.npy
|   |           |-- p_step_141.png
|   |           |-- p_step_142.npy
|   |           |-- p_step_142.png
|   |           |-- p_step_143.npy
|   |           |-- p_step_143.png
|   |           |-- p_step_144.npy
|   |           |-- p_step_144.png
|   |           |-- p_step_145.npy
|   |           |-- p_step_145.png
|   |           |-- p_step_146.npy
|   |           |-- p_step_146.png
|   |           |-- p_step_147.npy
|   |           |-- p_step_147.png
|   |           |-- p_step_148.npy
|   |           |-- p_step_148.png
|   |           |-- p_step_149.npy
|   |           |-- p_step_149.png
|   |           |-- p_step_150.npy
|   |           |-- p_step_150.png
|   |           |-- p_step_151.npy
|   |           |-- p_step_151.png
|   |           |-- p_step_152.npy
|   |           |-- p_step_152.png
|   |           |-- p_step_153.npy
|   |           |-- p_step_153.png
|   |           |-- p_step_154.npy
|   |           |-- p_step_154.png
|   |           |-- p_step_155.npy
|   |           `-- p_step_155.png
|   |-- toy_evidential_set_transformer.py
|   |   def collect_evidence_stats_per_step(alpha_t, beta_t, alpha_cum_t, beta_cum_t, stats, t)
|   |   def save_evidence_timeseries_plots(stats, out_dir, tag)
|   |   def save_checkpoint(path, epoch, model, opt, scaler, log_lines, hp_dict)
|   |   def load_checkpoint(path, device)
|   |   def _load_field_source_idx(npz_path)
|   |   def load_ert_npz(ert_npz_path)
|   |   def collate_fn(batch)
|   |   def compute_time_average_loss(model, sets, y, start_t, stride, chunk, kl_lambda, topk_lambda)
|   |   def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda, eps)
|   |   def run_train(epochs, n_train, n_val, save_dir, ckpt_path, resume)
|   |   def run_test_single_field(ckpt_path, out_dir, field_offset, first_output_step, last_output_step, save_alpha_beta, stamp)
|   |       class HParams:
|   |       class ERTOracleDataset(Dataset):
|   |   def __init__(self, ert_npz_path, oracle_npz_path, hp, pick_scale_index, max_fields, field_offset, warmup_npz_path, warmup_pick, active_select, active_seed, patterns_per_field)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class DeepSetsEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden_dim, agg)
|   |   def forward(self, X, mask)
|   |       class FiLMDecoder(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, dropout)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class EvidentialHead(nn.Module):
|   |   def __init__(self, in_dim, eps, emax)
|   |   def forward(self, U)
|   |       class ToyModel(nn.Module):
|   |   def __init__(self, hp, roi_idx)
|   |   def forward_time(self, S_t)
|   |   def forward_sequence(self, sets)
|   |   def forward_sequence_window(self, sets_window)
|   |   def forward_sequence_chunked(self, sets, chunk, start_t)
|   |-- toy_evidential_set_transformer1.py
|   |   def sinusoidal_position_embedding(x, dim)
|   |   def collate_fn(batch)
|   |   def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda, eps)
|   |   def run_train(epochs, n_train, n_val, save_dir)
|   |   def run_smoke(n_val)
|   |       class HParams:
|   |       class ToyERTDataset(Dataset):
|   |   def __init__(self, n_samples, hp, noise_std, seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class MultiheadSelfAttention(nn.Module):
|   |   def __init__(self, dim, num_heads, dropout)
|   |   def forward(self, x, key_padding_mask)
|   |       class PMA(nn.Module):
|   |   def __init__(self, dim, num_heads, num_seeds)
|   |   def forward(self, X, key_padding_mask)
|   |       class SetEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden_dim, heads, layers, pma_seeds, dropout)
|   |   def forward(self, X, mask)
|   |       class CrossAttentionDecoder(nn.Module):
|   |   def __init__(self, query_dim, kv_dim, hidden_dim, heads, layers, dropout)
|   |   def forward(self, queries, keys_values)
|   |       class EvidentialHead(nn.Module):
|   |   def __init__(self, in_dim, eps)
|   |   def forward(self, U)
|   |       class ToyModel(nn.Module):
|   |   def __init__(self, hp)
|   |   def forward_time(self, S_t, M_t, t_norm)
|   |   def forward_sequence(self, sets_list, masks_list)
|   |-- toy_evidential_set_transformer2.py
|   |   def sinusoidal_position_embedding(x, dim)
|   |   def collate_fn(batch)
|   |   def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda, eps)
|   |   def run_train(epochs, n_train, n_val, save_dir)
|   |   def run_smoke(n_val)
|   |       class HParams:
|   |       class ToyERTDataset(Dataset):
|   |   def __init__(self, n_samples, hp, noise_std, seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class MultiheadSelfAttention(nn.Module):
|   |   def __init__(self, d_model, heads, dropout)
|   |   def _ensure_key_mask(self, x, key_padding_mask)
|   |   def forward(self, x, key_padding_mask)
|   |       class PMA(nn.Module):
|   |   def __init__(self, d_model, k, heads, dropout)
|   |   def _ensure_key_mask(self, X, key_padding_mask)
|   |   def forward(self, X, key_padding_mask)
|   |       class SetEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden_dim, heads, layers, pma_seeds, dropout)
|   |   def forward(self, X, mask)
|   |       class CrossAttentionDecoder(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, dropout)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class SetDecoderSAB(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, layers)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class EvidentialHead(nn.Module):
|   |   def __init__(self, in_dim, eps)
|   |   def forward(self, U)
|   |       class ToyModel(nn.Module):
|   |   def __init__(self, hp)
|   |   def forward_time(self, S_t, M_t)
|   |   def forward_sequence(self, sets_list, masks_list)
|   |-- toy_evidential_set_transformer3.py
|   |   def sinusoidal_position_embedding(x, dim)
|   |   def collate_fn(batch)
|   |   def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda, eps)
|   |   def run_train(epochs, n_train, n_val, save_dir)
|   |   def run_smoke(n_val)
|   |       class HParams:
|   |       class ToyERTDataset(Dataset):
|   |   def __init__(self, n_samples, hp, noise_std, seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class MultiheadSelfAttention(nn.Module):
|   |   def __init__(self, d_model, heads, dropout)
|   |   def _ensure_key_mask(self, x, key_padding_mask)
|   |   def forward(self, x, key_padding_mask)
|   |       class PMA(nn.Module):
|   |   def __init__(self, d_model, k, heads, dropout)
|   |   def _ensure_key_mask(self, X, key_padding_mask)
|   |   def forward(self, X, key_padding_mask)
|   |       class SetEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden_dim, heads, layers, pma_seeds, dropout)
|   |   def forward(self, X, mask)
|   |       class CrossAttentionDecoder(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, dropout)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class SetDecoderSAB(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, layers)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class EvidentialHead(nn.Module):
|   |   def __init__(self, in_dim, eps)
|   |   def forward(self, U)
|   |       class ToyModel(nn.Module):
|   |   def __init__(self, hp)
|   |   def forward_time(self, S_t, M_t)
|   |   def forward_sequence(self, sets_list, masks_list)
|   |-- toy_evidential_set_transformer4.py
|   |   def _load_field_source_idx(npz_path)
|   |   def load_ert_npz(ert_npz_path)
|   |   def sinusoidal_position_embedding(x, dim)
|   |   def collate_fn(batch)
|   |   def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda, eps)
|   |   def run_train(epochs, n_train, n_val, save_dir)
|   |   def run_smoke(n_val)
|   |       class HParams:
|   |       class ToyERTDataset(Dataset):
|   |   def __init__(self, n_samples, hp, noise_std, seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class ERTOracleDataset(Dataset):
|   |   def __init__(self, ert_npz_path, oracle_npz_path, hp, pick_scale_index, max_fields, field_offset, warmup_npz_path, warmup_pick)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class MultiheadSelfAttention(nn.Module):
|   |   def __init__(self, d_model, heads, dropout)
|   |   def _ensure_key_mask(self, x, key_padding_mask)
|   |   def forward(self, x, key_padding_mask)
|   |       class PMA(nn.Module):
|   |   def __init__(self, d_model, k, heads, dropout)
|   |   def _ensure_key_mask(self, X, key_padding_mask)
|   |   def forward(self, X, key_padding_mask)
|   |       class SetEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden_dim, heads, layers, pma_seeds, dropout)
|   |   def forward(self, X, mask)
|   |       class CrossAttentionDecoder(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, dropout)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class SetDecoderSAB(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, layers)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class EvidentialHead(nn.Module):
|   |   def __init__(self, in_dim, eps)
|   |   def forward(self, U)
|   |       class ToyModel(nn.Module):
|   |   def __init__(self, hp)
|   |   def forward_time(self, S_t, M_t)
|   |   def forward_sequence(self, sets_list, masks_list)
|   |-- toy_evidential_set_transformer5.py
|   |   def _load_field_source_idx(npz_path)
|   |   def load_ert_npz(ert_npz_path)
|   |   def sinusoidal_position_embedding(x, dim)
|   |   def collate_fn(batch)
|   |   def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda, eps)
|   |   def run_train(epochs, n_train, n_val, save_dir)
|   |   def run_smoke(n_val)
|   |       class HParams:
|   |       class ToyERTDataset(Dataset):
|   |   def __init__(self, n_samples, hp, noise_std, seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class ERTOracleDataset(Dataset):
|   |   def __init__(self, ert_npz_path, oracle_npz_path, hp, pick_scale_index, max_fields, field_offset, warmup_npz_path, warmup_pick, active_select, active_seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class MultiheadSelfAttention(nn.Module):
|   |   def __init__(self, d_model, heads, dropout)
|   |   def _ensure_key_mask(self, x, key_padding_mask)
|   |   def forward(self, x, key_padding_mask)
|   |       class PMA(nn.Module):
|   |   def __init__(self, d_model, k, heads, dropout)
|   |   def _ensure_key_mask(self, X, key_padding_mask)
|   |   def forward(self, X, key_padding_mask)
|   |       class SetEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden_dim, heads, layers, pma_seeds, dropout)
|   |   def forward(self, X, mask)
|   |       class CrossAttentionDecoder(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, dropout)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class SetDecoderSAB(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, layers)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class EvidentialHead(nn.Module):
|   |   def __init__(self, in_dim, eps)
|   |   def forward(self, U)
|   |       class ToyModel(nn.Module):
|   |   def __init__(self, hp, roi_idx)
|   |   def forward_time(self, S_t, M_t)
|   |   def forward_sequence(self, sets_list, masks_list)
|   |-- toy_evidential_set_transformer6.py
|   |   def _load_field_source_idx(npz_path)
|   |   def load_ert_npz(ert_npz_path)
|   |   def collate_fn(batch)
|   |   def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda, eps)
|   |   def run_train(epochs, n_train, n_val, save_dir)
|   |   def run_smoke(n_val)
|   |       class HParams:
|   |       class ToyERTDataset(Dataset):
|   |   def __init__(self, n_samples, hp, noise_std, seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class ERTOracleDataset(Dataset):
|   |   def __init__(self, ert_npz_path, oracle_npz_path, hp, pick_scale_index, max_fields, field_offset, warmup_npz_path, warmup_pick, active_select, active_seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class MultiheadSelfAttention(nn.Module):
|   |   def __init__(self, d_model, heads, dropout)
|   |   def _ensure_key_mask(self, x, key_padding_mask)
|   |   def forward(self, x, key_padding_mask)
|   |       class PMA(nn.Module):
|   |   def __init__(self, d_model, k, heads, dropout)
|   |   def _ensure_key_mask(self, X, key_padding_mask)
|   |   def forward(self, X, key_padding_mask)
|   |       class SetEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden_dim, heads, layers, pma_seeds, dropout)
|   |   def forward(self, X, mask)
|   |       class CrossAttentionDecoder(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, dropout)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class EvidentialHead(nn.Module):
|   |   def __init__(self, in_dim, eps)
|   |   def forward(self, U)
|   |       class ToyModel(nn.Module):
|   |   def __init__(self, hp, roi_idx)
|   |   def forward_time(self, S_t, M_t)
|   |   def forward_sequence(self, sets_list, masks_list)
|   |-- toy_evidential_set_transformer7.py
|   |   def _load_field_source_idx(npz_path)
|   |   def load_ert_npz(ert_npz_path)
|   |   def collate_fn(batch)
|   |   def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda, eps)
|   |   def run_train(epochs, n_train, n_val, save_dir)
|   |   def run_smoke(n_val)
|   |   def run_test_single_field(ckpt_path, out_dir, field_offset, first_output_step, last_output_step, save_alpha_beta)
|   |       class HParams:
|   |       class ToyERTDataset(Dataset):
|   |   def __init__(self, n_samples, hp, noise_std, seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class ERTOracleDataset(Dataset):
|   |   def __init__(self, ert_npz_path, oracle_npz_path, hp, pick_scale_index, max_fields, field_offset, warmup_npz_path, warmup_pick, active_select, active_seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class MultiheadSelfAttention(nn.Module):
|   |   def __init__(self, d_model, heads, dropout)
|   |   def _ensure_key_mask(self, x, key_padding_mask)
|   |   def forward(self, x, key_padding_mask)
|   |       class PMA(nn.Module):
|   |   def __init__(self, d_model, k, heads, dropout)
|   |   def _ensure_key_mask(self, X, key_padding_mask)
|   |   def forward(self, X, key_padding_mask)
|   |       class SetEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden_dim, heads, layers, pma_seeds, dropout)
|   |   def forward(self, X, mask)
|   |       class CrossAttentionDecoder(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, dropout)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class EvidentialHead(nn.Module):
|   |   def __init__(self, in_dim, eps)
|   |   def forward(self, U)
|   |       class ToyModel(nn.Module):
|   |   def __init__(self, hp, roi_idx)
|   |   def forward_time(self, S_t, M_t)
|   |   def forward_sequence(self, sets_list, masks_list)
|   |-- toy_evidential_set_transformer8.py
|   |   def _load_field_source_idx(npz_path)
|   |   def load_ert_npz(ert_npz_path)
|   |   def collate_fn(batch)
|   |   def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda, eps)
|   |   def run_train(epochs, n_train, n_val, save_dir)
|   |   def run_test_single_field(ckpt_path, out_dir, field_offset, first_output_step, last_output_step, save_alpha_beta, stamp)
|   |       class HParams:
|   |       class ERTOracleDataset(Dataset):
|   |   def __init__(self, ert_npz_path, oracle_npz_path, hp, pick_scale_index, max_fields, field_offset, warmup_npz_path, warmup_pick, active_select, active_seed)
|   |   def __len__(self)
|   |   def __getitem__(self, idx)
|   |       class MultiheadSelfAttention(nn.Module):
|   |   def __init__(self, d_model, heads, dropout)
|   |   def _ensure_key_mask(self, x, key_padding_mask)
|   |   def forward(self, x, key_padding_mask)
|   |       class PMA(nn.Module):
|   |   def __init__(self, d_model, k, heads, dropout)
|   |   def _ensure_key_mask(self, X, key_padding_mask)
|   |   def forward(self, X, key_padding_mask)
|   |       class SetEncoder(nn.Module):
|   |   def __init__(self, in_dim, hidden_dim, heads, layers, pma_seeds, dropout)
|   |   def forward(self, X, mask)
|   |       class CrossAttentionDecoder(nn.Module):
|   |   def __init__(self, query_dim, ctx_dim, hidden_dim, heads, dropout)
|   |   def forward(self, pixel_queries, ctx_vec)
|   |       class EvidentialHead(nn.Module):
|   |   def __init__(self, in_dim, eps)
|   |   def forward(self, U)
|   |       class ToyModel(nn.Module):
|   |   def __init__(self, hp, roi_idx)
|   |   def forward_time(self, S_t)
|   |   def forward_sequence(self, sets)
|   |-- toy_test.ipynb
|   |-- train_ert_surrogate.py
|   |   def load_npz_dataset(path, z_dim, dn_dim)
|   |   def _infer_rows_per_field_from_Z(Z, atol)
|   |   def split_by_field(N, meta, seed, frac, *, n_ab, n_mn_per_ab, rows_per_field, Z)
|   |   def standardize_train_only(Z, idx_train)
|   |   def rmse(y_true, y_pred)
|   |   def eval_epoch(model, loader, device)
|   |   def make_criterion(args)
|   |   def train_loop(model, opt, loaders, device, criterion, epochs, patience, sched, ckpt_path, amp, start_epoch, init_best, init_noimp, init_hist, resume_ckpt_path)
|   |   def main()
|   |       class ResidualHead(nn.Module):
|   |   def __init__(self, in_dim, hidden, dropout)
|   |   def forward(self, x)
|   |       class SurrogateWithResidual(nn.Module):
|   |   def __init__(self, base, res_head, use_mu_as_feat)
|   |   def forward(self, x)
|   |       class SurrogateFiLM(nn.Module):
|   |   def __init__(self, k_lat, d_feat, hidden, film_hidden, dropout)
|   |   def forward(self, x)
|   |       class RowsDataset(Dataset):
|   |   def __init__(self, X, y)
|   |   def __len__(self)
|   |   def __getitem__(self, i)
|   |       class MLP(nn.Module):
|   |   def __init__(self, in_dim, hidden, dropout)
|   |   def forward(self, x)
|   |-- verify_alignment.py
|   |   def load_meta(ds)
|   |   def main()
|   `-- y_example_field0.png
|-- reports\
|   `-- .gitkeep
|-- requirements.txt
|-- scripts\
|   |-- 01_make_fields.py
|   |   def build_cli_args(cfg)
|   |   def main()
|   |-- 02_fit_pca_and_project.py
|   |   def kv_to_cli(cfg)
|   |   def main()
|   |-- 03_make_surrogate_pairs_pygimli.py
|   |   def _append_flag(args, flag, value)
|   |   def cfg_to_cli(cfg)
|   |   def main()
|   |-- 03_make_surrogate_pairs_wenner.py
|   |   def build_argv_from_yaml(cfg, script_path)
|   |   def main()
|   |-- 04_make_oracle_masks.py
|   |   def nested_get(d, path)
|   |   def main()
|   |-- 04b_reduce_oracle_diversity.py
|   |   def main()
|   |-- 05_train_surrogate.py
|   |   def main()
|   `-- 06_eval_surrogate.py
|       def load_module_from_path(module_name, path)
|       def main()
`-- src\
    `-- go_boed_ert\
```
