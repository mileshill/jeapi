#!/usr/bin/env bash

# timestamp_results.sh
# Locates any CSV, XLSX files in current directory and adds timestamp.
# Movies files to the Dropbox directory.
# OLD -> FILENAME
# NEW -> YYYYMMDD_FILENAME


# Launch the recipe calculations
python main.py

# Run the JSON builder script.
# The JSON will be consumed by the dashboard
python result_csv_to_json.py

# Locate and rename
TIMESTAMP=$(date +'%Y%m%d')
for fname in $(find . -maxdepth 1 -name "*.csv" -o -name "*.xlsx") 
do
    FILENAME=$(basename $fname)
    NEW=$TIMESTAMP'_'$FILENAME
    mv $FILENAME ~/Dropbox/iCAP_Project/Results/Analysis/$NEW
done

