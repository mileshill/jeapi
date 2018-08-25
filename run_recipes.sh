#!/usr/bin/env bash

# Locates any CSV files in current directory and adds timestamp.
# Moves files to the Dropbox directory.
# OLD -> FILENAME
# NEW -> YYYYMMDD_FILENAME

TIMESTAMP=$(date +'%Y_%m_%d')
# Move into project directory
echo "cd /home/ubuntu/JustEnergy"
cd /home/ubuntu/JustEnergy

# Launch the recipe calculations
echo "$(date +'%c') Recipe calculations started."
/home/ubuntu/miniconda3/envs/predictions/bin/python3.5  main.py

# Run the JSON builder script.
# The JSON will be consumed by the dashboard
echo "$(date +'%c') Launching <result_csv_to_json.py>"
/home/ubuntu/miniconda3/envs/predictions/bin/python3.5 result_csv_to_json.py
# The `result_csv_to_json` will create the ./data directory and populate it. 
#  This directory structure is required by the Dashboard tool.
echo "$(date +'c') Update ./data directory for Dashbaord"
cp -r data /home/ubuntu/Dashboard/scripts/
cp -r data /var/www/html/scripts/


echo "$(date +'%c') Moving CSV/JSON into /home/ubuntu/archives/${TIMESTAMP}"
# Locate and rename all newly created recipe CSV. Timestamps are prepended to the file name
#for fname in $(find . -maxdepth 1 -name "*.csv" -o -name "*.json") 
#do
#    FILENAME=$(basename $fname)
#    NEW=$TIMESTAMP'_'$FILENAME
#    mv $FILENAME $NEW
#done

# Move remaining files to /home/ubuntu/archives
if [ ! -d /home/ubuntu/archives/${TIMESTAMP} ]; then
	mkdir -p /home/ubuntu/archives/${TIMESTAMP}
fi
mv *csv /home/ubuntu/archives/${TIMESTAMP}
mv *json /home/ubuntu/archives/${TIMESTAMP}

echo "$(date +'%c') Copying files into S3"
chown -R ubuntu:ubuntu /home/ubuntu/archives/${TIMESTAMP}
cd /home/ubuntu/archives/${TIMESTAMP}
#/home/ubuntu/miniconda3/bin/aws s3 --profile business_labs cp ${TIMESTAMP}_recipe.csv s3://just-energy-capacity/results/
#/home/ubuntu/miniconda3/bin/aws s3 --profile business_labs cp ${TIMESTAMP}_premise_explorer.json s3://just-energy-capacity/results/ 

/home/ubuntu/miniconda3/bin/aws s3 --profile business_labs cp recipe.csv s3://just-energy-capacity/results/
/home/ubuntu/miniconda3/bin/aws s3 --profile business_labs cp premise_explorer.json s3://just-energy-capacity/results/ 

# Copy NITS
/home/ubuntu/miniconda3/bin/aws s3 --profile business_labs cp peco_interval_nits.csv s3://just-energy-capacity/results/
/home/ubuntu/miniconda3/bin/aws s3 --profile business_labs cp peco_demand_nits.csv s3://just-energy-capacity/results/
