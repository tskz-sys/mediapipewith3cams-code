import argparse, json, os
import numpy as np

def expand_poly(poly, scale):
    poly = np.asarray(poly, dtype=float)
    c = poly.mean(axis=0)  # centroid (簡易)
    return (c + scale * (poly - c)).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--scale", type=float, required=True, help="1.0=変更なし, 1.2=20%拡大 など")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "polygon_uv" in data:
        data["polygon_uv"] = expand_poly(data["polygon_uv"], args.scale)

    if isinstance(data.get("polygons"), list):
        for entry in data["polygons"]:
            if isinstance(entry, dict) and "polygon_uv" in entry:
                entry["polygon_uv"] = expand_poly(entry["polygon_uv"], args.scale)

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Wrote:", args.output_json)

if __name__ == "__main__":
    main()
