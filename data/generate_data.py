"""Recreate the individual CSV: python data/generate_data.py --student-id ug2424671@ines.ac.rw --seed 4671"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation import generate_dataset, write_dataset, write_fingerprint  # noqa: E402
from src.validation import parse_identity  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-id", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    ident = parse_identity(a.student_id, a.seed)
    csv = ROOT / "data" / f"AI_A2_{ident['reg_digits']}.csv"
    write_dataset(generate_dataset(ident["seed"]), csv)
    digest = write_fingerprint(csv, ROOT / "artifacts" / "data_fingerprint.txt",
                               ident["verification_code"])
    print(f"{ident['verification_code']} | wrote {csv.name} | sha256 {digest}")


if __name__ == "__main__":
    main()
