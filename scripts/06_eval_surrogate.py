# scripts/06_eval_surrogate.py
import argparse, json, yaml, sys
from pathlib import Path
import importlib.util

def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    ap = argparse.ArgumentParser(description="Evaluate trained surrogate (thin CLI).")
    ap.add_argument("--config", "-c", required=True, help="Path to eval_surrogate.yml")
    ap.add_argument(
        "--script",
        default=str(Path("./build/eval_surrogate.py")),
        help="Path to build/eval script (default: ./build/eval_surrogate.py)",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    script_path = Path(args.script).resolve()

    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}", file=sys.stderr); sys.exit(2)
    if not script_path.exists():
        print(f"[ERROR] Build script not found: {script_path}", file=sys.stderr); sys.exit(2)

    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    mod = load_module_from_path("eval_surrogate_build", script_path)

    if not hasattr(mod, "run_eval"):
        print(f"[ERROR] The build script at {script_path} does not define run_eval(cfg).", file=sys.stderr)
        sys.exit(2)

    summary, outdir = mod.run_eval(cfg)

    # Save and print a summary
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
