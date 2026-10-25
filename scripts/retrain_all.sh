#!/bin/bash
# Reruns EVERY training of the paper (level 2 of docs/03_REPRODUCING_THE_PAPER.md).
# Writes to results/rerun/; the original results in results/ are never touched.
#
# Usage (environment active: conda activate ltveg; images in data/dataset/):
#     bash scripts/retrain_all.sh              # every stage, in sequence (~4-6 h)
#     bash scripts/retrain_all.sh resnet yolo  # selected stages only
#
# Stages: single resnet ablation yolo yolo_auto yolo_nano cost compare
# Bootstraps are resumable: rerunning skips replicates already saved.
set -e
cd "$(dirname "$0")/.."

DATA=data/dataset
OUT=results/rerun
mkdir -p "$OUT"
STAGES=${*:-"single resnet ablation yolo yolo_auto yolo_nano cost compare"}

# prevent the machine from sleeping during long runs, when available (Linux/systemd)
if command -v systemd-inhibit >/dev/null 2>&1 && [ -z "$LTVEG_INHIBITED" ]; then
    export LTVEG_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --why="LTVeg: retraining" bash "$0" "$@"
fi

stage() { echo; echo "=== [$1] start $(date) ==="; }

for e in $STAGES; do
    case $e in
    single)
        stage single
        python src/training/train_baseline.py --data-dir $DATA --out-dir $OUT/train_baseline ;;
    resnet)
        stage resnet
        python src/training/bootstrap_resnet.py --data-dir $DATA --out-dir $OUT/bootstrap_resnet --n 30 ;;
    ablation)
        stage ablation
        python src/training/yolo_ablation.py --data-dir $DATA --out-dir $OUT/yolo_ablation ;;
    yolo)
        stage yolo
        python src/training/bootstrap_yolo.py --data-dir $DATA --out-dir $OUT/bootstrap_yolo_adamw --n 30 --config explicit_adamw ;;
    yolo_auto)
        stage yolo_auto
        python src/training/bootstrap_yolo.py --data-dir $DATA --out-dir $OUT/bootstrap_yolo_auto --n 30 --config baseline_current ;;
    yolo_nano)
        stage yolo_nano
        python src/training/bootstrap_yolo.py --data-dir $DATA --out-dir $OUT/bootstrap_yolo_nano --n 30 --config light_aug_adamw_nano ;;
    cost)
        stage cost
        python src/training/compute_cost.py --data-dir $DATA --out $OUT/compute_cost.json ;;
    compare)
        stage compare
        python src/analysis/bootstrap_compare.py --resnet-dir $OUT/bootstrap_resnet \
            --yolo-dir $OUT/bootstrap_yolo_adamw --out $OUT/bootstrap_comparison.json ;;
    *)
        echo "unknown stage: $e"; exit 1 ;;
    esac
done
echo; echo "=== all done $(date) -- results in $OUT/ ==="
