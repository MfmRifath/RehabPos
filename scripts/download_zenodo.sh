#!/usr/bin/env bash
# Download all files of a public Zenodo record into data/raw/<name>/ and verify checksums.
# NEVER run automatically: dataset downloads need the owner's explicit approval
# (project rule). Check the dataset's license on its Zenodo page first.
#
# Usage: scripts/download_zenodo.sh <record_id> <dataset_name> --yes
#   e.g. scripts/download_zenodo.sh 13305826 rehab24_6 --yes        (REHAB24-6)
#        scripts/download_zenodo.sh 4610859 intellirehab --yes      (IntelliRehabDS; confirm the
#                                                                    record id behind the DOI)
set -euo pipefail

if [[ $# -ne 3 || "$3" != "--yes" ]]; then
  echo "usage: $0 <zenodo_record_id> <dataset_name> --yes" >&2
  echo "Refusing to download without --yes (owner approval required)." >&2
  exit 2
fi
record="$1"; name="$2"
dest="data/raw/${name}"
mkdir -p "$dest"

meta="$(curl -fsSL "https://zenodo.org/api/records/${record}")"
python3 - "$meta" > "$dest/.zenodo_files.tsv" <<'PY'
import json, sys
rec = json.loads(sys.argv[1])
md = rec.get("metadata", {})
print(f"# {md.get('title', '?')} | license: {md.get('license', {}).get('id', '?')}", file=sys.stderr)
total = 0
for f in rec.get("files", []):
    total += f.get("size", 0)
    print(f"{f['key']}\t{f['links']['self']}\t{f.get('checksum', '')}")
print(f"# {len(rec.get('files', []))} files, {total / 1e6:.1f} MB total", file=sys.stderr)
PY

while IFS=$'\t' read -r key url checksum; do
  out="$dest/$key"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" ]]; then echo "exists: $out"; continue; fi
  echo "downloading $key"
  curl -fL --retry 3 -o "$out.part" "$url"
  if [[ "$checksum" == md5:* ]]; then
    got="$(md5 -q "$out.part" 2>/dev/null || md5sum "$out.part" | cut -d' ' -f1)"
    [[ "$got" == "${checksum#md5:}" ]] || { echo "checksum mismatch for $key" >&2; exit 1; }
  fi
  mv "$out.part" "$out"
done < "$dest/.zenodo_files.tsv"
echo "done -> $dest (extract archives there if needed; data/ is git-ignored)"
