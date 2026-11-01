#!/usr/bin/env bash
# Upper-limb stroke rehab exercise video dataset (Data in Brief 2026), Mendeley Data
# doi:10.17632/49h9dcwx5v.1, CC BY 4.0.
# Mendeley Data downloads are done from the dataset page ("Download All"); this script only
# prints instructions and creates the target folder. It downloads nothing.
set -euo pipefail
dest="data/raw/dib_upper"
cat <<MSG
1. Open https://data.mendeley.com/datasets/49h9dcwx5v/1 and read the license (CC BY 4.0).
2. Click "Download All" and unzip into: $dest/
3. Run: uv run rehabpose data stats dib_upper --i-accept-license
   The expected layout is ASSUMED ($dest/<exercise>/<complete|incomplete>/<video>);
   if it differs, adjust configs/datasets/dib_upper.yaml.
MSG
mkdir -p "$dest"
