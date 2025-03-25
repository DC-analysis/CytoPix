#!/bin/bash
# Call this script whenever the .ui files changed.
cd "$(dirname "$0")"
cd ..

FILES=$(ls -1 ./ui/*.ui)

for f in $FILES
do
    filename=$(basename -- "$f")
    outname="${filename%.*}_ui.py"
    # disable flake8 for these files
    echo "# flake8: noqa" > "./src/cytopix/${outname}"
    # populate the file
    pyuic6 $f >> "./src/cytopix/${outname}"
done
