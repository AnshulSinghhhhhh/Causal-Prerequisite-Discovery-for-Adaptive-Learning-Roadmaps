#!/usr/bin/env bash
# LightGAP dataset downloader (build order step 1).
#
# Pulls the Module 1/2 training + validation corpora (Section 4) into
# data/external/. The PnPR-GCN Graph Split is the leakage-safe 5-fold CV split
# of AL-CPL and must be used INSTEAD of a naive random split (transitively
# implied test edges would otherwise leak from training).
#
# Usage:  bash scripts/download_datasets.sh [--refd]
#   --refd   also fetch RefD (only if you intend to re-test that hypothesis;
#            Section 12 notes it previously REGRESSED results on this data).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/data/external"
mkdir -p "$EXT"

WITH_REFD=0
for arg in "$@"; do
  case "$arg" in
    --refd) WITH_REFD=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

clone() {
  local name="$1" url="$2" dest="$3"
  if [ -d "$dest/.git" ]; then
    echo "[skip] $name already present at $dest"
  else
    echo "[clone] $name -> $dest"
    git clone --depth 1 "$url" "$dest"
  fi
}

clone "AL-CPL"                     "https://github.com/harrylclc/AL-CPL-dataset"                 "$EXT/al-cpl"
clone "PnPR-GCN Graph Split"       "https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24"           "$EXT/pnpr-gcn"
clone "LectureBank (LectureBankCD)" "https://github.com/Yale-LILY/LectureBank"                  "$EXT/lecturebank"
clone "University Course Prereq"   "https://github.com/harrylclc/eaai17-cpr-recover"            "$EXT/university-course"

if [ "$WITH_REFD" -eq 1 ]; then
  clone "RefD" "https://github.com/harrylclc/RefD-dataset" "$EXT/refd"
fi

echo ""
echo "Done. Verify the PnPR-GCN split is present:"
echo "  ls \"$EXT/pnpr-gcn/Graph_Split\""
echo ""
echo "Next: run scripts/run_full_pipeline.py (requires network for embeddings/model weights)."