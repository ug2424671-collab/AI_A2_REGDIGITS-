"""Score one farm record: python predict.py --json samples/valid.json"""
import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from src.inference import ModelsNotFoundError, predict_record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True, type=Path, help="path to one JSON farm record")
    a = p.parse_args()
    try:
        record = json.loads(a.json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read JSON record: {exc}", file=sys.stderr)
        return 2
    try:
        result = predict_record(record)
    except ValidationError as exc:
        errors = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
                  for e in exc.errors()]
        print(json.dumps({"error": "invalid input", "details": errors}, indent=2),
              file=sys.stderr)
        return 2
    except ModelsNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
