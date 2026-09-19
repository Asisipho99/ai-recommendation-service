"""Run the whole data + ML pipeline from the command line.

Useful for verifying everything offline without starting Flask:

    python scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import eda, loader, prepare  # noqa: E402
from ml import registry  # noqa: E402


def show(title: str) -> None:
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


def main() -> int:
    show("1. SOURCE")
    info = loader.source_info()
    print(json.dumps(info, indent=2))
    if not info["available"]:
        print("No dataset. Run: python scripts/download_dataset.py")
        return 1

    show("2. LOAD RAW")
    raw = loader.load_raw()
    print(f"rows    : {len(raw):,}")
    print(f"columns : {list(raw.columns)}")

    show("3. EDA - MISSING VALUES")
    for row in eda.missing_values(raw):
        if row["missing"]:
            print(f"  {row['column']:<15} {row['missing']:>9,} missing "
                  f"({row['missing_pct']:>5.2f}%)")
    print("  (columns with no missing values omitted)")

    show("4. EDA - DATA QUALITY FINDINGS")
    for finding in eda.quality_findings(raw):
        print(f"  {finding['issue']}")
        print(f"    rows: {finding['affected_rows']:,} "
              f"({finding['affected_pct']}%)  -> {finding['treatment']}")

    show("5. CLEANING")
    cleaned, steps = prepare.clean(raw)
    for step in steps:
        if step["step"] == "TOTAL":
            print(f"  {'TOTAL':<45} {step['rows_after']:>9,} rows retained "
                  f"({step['retained_pct']}%)")
        else:
            print(f"  {step['step']:<45} -{step['rows_removed']:>8,}")

    show("6. FEATURE ENGINEERING")
    features, meta = prepare.build_customer_features(cleaned)
    print(f"customers          : {meta['customers']:,}")
    print(f"features           : {meta['feature_columns']}")
    print(f"label              : {meta['label_column']}")
    print(f"feature window end : {meta['feature_window_end']}")
    print(f"positive class     : {meta['positive_label_count']:,} "
          f"({meta['positive_label_pct']}%)")

    prepare.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(prepare.FEATURES_FILE, index=False, compression="gzip")
    print(f"written            : {prepare.FEATURES_FILE.name}")

    show("7. TRAIN AND COMPARE")
    report = registry.train_all()

    print(f"{'algorithm':<22} {'accuracy':>9} {'precision':>10} "
          f"{'recall':>8} {'f1':>8} {'roc_auc':>8}")
    print("-" * 70)
    for row in report["supervised_comparison"]:
        auc = row["roc_auc"] if row["roc_auc"] is not None else 0
        print(f"{row['algorithm']:<22} {row['accuracy']:>9.4f} "
              f"{row['precision']:>10.4f} {row['recall']:>8.4f} "
              f"{row['f1']:>8.4f} {auc:>8.4f}")

    seg = report["segmentation"]
    print()
    print(f"{seg['algorithm']:<22} silhouette {seg['metrics']['silhouette']} "
          f"at k={seg['chosen_k']}")
    for cluster in seg["clusters"]:
        print(f"    cluster {cluster['cluster']}: {cluster['customers']:>5,} "
              f"customers  {cluster['label']}")

    print()
    print(f"best supervised : {report['best_supervised']['algorithm']}")
    print(f"algorithms      : {report['algorithm_count']} "
          f"({', '.join(report['algorithms'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
