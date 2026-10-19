"""Single entry point: python run_all.py --student-id ug2424671@ines.ac.rw --seed 4671"""
import argparse
import sys
from pathlib import Path

from src.config import PROJECT_ROOT
from src.pipeline import run_pipeline


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-id", required=True, help="institutional username, e.g. ug2424671@ines.ac.rw")
    p.add_argument("--seed", type=int, required=True, help="last four registration digits (LLLL)")
    a = p.parse_args()
    try:
        s = run_pipeline(a.student_id, a.seed, PROJECT_ROOT)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    reg, cls, clu = s["regression"], s["classification"], s["clustering"]
    sel_r, sel_c = reg["selected_model"], cls["selected_model"]
    print(f"Verification code : {s['identity']['verification_code']}")
    print(f"Dataset           : {s['csv']} ({s['raw_rows']} raw rows, {s['clean_rows']} after "
          f"removing {s['duplicates']} duplicates; missing={s['missing']})")
    print(f"SHA-256           : {s['fingerprint']}")
    print(f"Regression        : {sel_r}  MAE={reg['candidates'][sel_r]['mae']:.4f} "
          f"RMSE={reg['candidates'][sel_r]['rmse']:.4f} R2={reg['candidates'][sel_r]['r2']:.4f}")
    c = cls["candidates"][sel_c]
    print(f"Classification    : {sel_c}  P={c['precision']:.3f} R={c['recall']:.3f} "
          f"F1={c['f1']:.3f} AUC={c['roc_auc']:.3f}")
    print(f"Clustering        : K={clu['selected_k']} (silhouette "
          f"{clu['silhouette_by_k'][str(clu['selected_k'])]:.4f})")
    print("Artifacts generated:")
    for f in sorted(p for d in ("artifacts", "models") for p in (PROJECT_ROOT / d).rglob("*")
                    if p.is_file()):
        print("  -", f.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
