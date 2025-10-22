# scripts/05_train_surrogate.py
# Thin CLI: read a YAML config and launch a training script with equivalent args.
import argparse, sys, yaml, shlex, subprocess
from pathlib import Path

# Map YAML keys to CLI flags used by the training script
KEYMAP = {
    # required
    "ds": "--ds",
    "out": "--out",

    # splits & seed
    "val": "--val",
    "test": "--test",
    "seed": "--seed",

    # booleans
    "use_dnorm": "--use-dnorm",
    "reciprocity_aug": "--reciprocity-aug",
    "amp": "--amp",
    "residual_head": "--residual-head",
    "res_use_mu_as_feat": "--res-use-mu-as-feat",

    # loader
    "workers": "--workers",
    "batch": "--batch",

    # optim
    "epochs": "--epochs",
    "patience": "--patience",
    "lr": "--lr",
    "weight_decay": "--weight-decay",

    # model
    "hidden": "--hidden",
    "dropout": "--dropout",

    # loss
    "loss": "--loss",
    "huber_delta": "--huber-delta",

    # residual extras
    "res_hidden": "--res-hidden",
    "res_dropout": "--res-dropout",
    "res_epochs": "--res-epochs",
    "res_lr": "--res-lr",
    "res_weight_decay": "--res-weight-decay",
    "res_huber_delta": "--res-huber-delta",

    # resume
    "resume": "--resume",

    # optional shape hints
    "z_dim": "--z-dim",
    "dn_dim": "--dn-dim",

    # optional dataset schema hints
    "n_ab": "--n-ab",
    "n_mn_per_ab": "--n-mn-per-ab",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="Path to train_surrogate.yml")
    ap.add_argument(
        "--script",
        default=str(Path("./build/train_ert_surrogate.py")),
        help="Path to training script (default: ./build/train_surrogate.py)",
    )
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("Config must be a YAML mapping")

    cmd = [args.python, args.script]

    # Walk the mapping in a stable order of insertion defined above
    for key, flag in KEYMAP.items():
        if key not in cfg or cfg[key] is None:
            continue
        val = cfg[key]

        # Booleans -> flags without value (only when True)
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
            continue

        # Lists/tuples -> append flag then all values (nargs='+')
        if isinstance(val, (list, tuple)):
            cmd.append(flag)
            cmd.extend(map(str, val))
            continue

        # Scalars
        cmd.extend([flag, str(val)])

    print("[runner] Exec:", shlex.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
