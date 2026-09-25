#!/bin/sh
# Converts everything in ./input into ./output (or the files you pass in).
cd "$(dirname "$0")"
python3 -c "import PIL, numpy" 2>/dev/null || python3 -m pip install -r requirements.txt
python3 skate3tex.py "$@"
