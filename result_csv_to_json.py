#!/usr/bin/env python3
from premise_explorer_structure_modifier import Reconstruct
import pandas as pd
import glob
import json
import numpy as np
# Path is relative to current directory
# Import all CSV files
# path = '/home/miles/Dropbox/iCAP_Project/Results/Analysis'
# file_names = glob.glob(path + '/[0-9]*.csv')
file_names = glob.glob('*rec*csv')

# Load all CSV files into memory
# Concatenate into single DataFrame
df = pd.DataFrame()
list_ = []
for file_ in file_names:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
temp = pd.concat(list_)



# Calculate the group statistics; based on PremiseId
grp = temp.groupby('PremiseId')
mad = grp['RecipeICap'].transform(lambda x: x.mad())
pct = grp['RecipeICap'].transform(lambda x: x.pct_change() * 100)


temp['TagVariability'] = mad
temp['PercentChange'] = pct.replace(to_replace=[np.nan, np.infty, -np.infty], value='null')





# Fill all NA values with something friendly to JavaScript
# temp.dropna(subset=['HistoricalICap'])
temp.fillna(value='null', inplace=True)

# Take only those Premises with 2 or more years history
#temp = temp.groupby('PremiseId').filter(lambda x: len(x) > 1)

# Convert year to numeric
temp['Year'] = pd.to_numeric(temp['Year'].copy(), downcast='float', errors='coerce')

# Take only those records where the ICap with computed by recipe
temp = temp[temp['RecipeICap'] != 'null']

# Main loop. Creates the nested JSON structure required by the front end
container = {}
for iso_util_prem_year, record in temp.groupby(
        ['ISO', 'UtilityId', 'PremiseId', 'Year']):
    iso, utility, premise, year = iso_util_prem_year

    year = str(int(year))

    # iso level
    if iso not in container.keys():
        container[iso] = {}
    iso_d = container[iso]

    # utility
    if utility not in iso_d.keys():
        iso_d[utility] = {'aggregate': {}, 'records': {}}
    util_agg_d = iso_d[utility]['aggregate']
    util_rec_d = iso_d[utility]['records']

    # update utility meta data
    if year not in util_agg_d.keys():
        util_agg_d[year] = {'min': 0., 'max': 0., 'count': 0., 'total': 0}

    util_year_d = util_agg_d[year]

    recipe_value = record.RecipeICap.values[0]
    util_year_d['min'] = min([util_year_d['min'], recipe_value])
    util_year_d['max'] = max([util_year_d['max'], recipe_value])
    util_year_d['total'] += recipe_value
    util_year_d['count'] += 1

    # premise
    if premise not in util_rec_d.keys():
        util_rec_d[premise] = {'metadata': {
            'min': (0, 1000000), 'max': (0, 0.), 'count': 0., 'total': 0}}
    prem_d = util_rec_d[premise]
    prem_agg_d = util_rec_d[premise]['metadata']

    meta_cols = ['ISO', 'MeterType', 'RateClass', 'Strata', 'UtilityId', 'PremiseId',
                 'TagVariability']
    prem_meta_d = record[meta_cols].to_dict(orient='records')[0]

    prem_agg_d['min'] = min(
        [prem_agg_d['min'], (year, recipe_value)], key=lambda tup: tup[1])
    prem_agg_d['max'] = max(
        [prem_agg_d['max'], (year, recipe_value)], key=lambda tup: tup[1])
    prem_agg_d['total'] += recipe_value
    prem_agg_d['count'] += 1

    for k, v in prem_meta_d.items():
        if k not in prem_agg_d.keys():
            prem_agg_d[k] = v

    # year
    if year not in prem_d.keys():
        prem_d[year] = {}
    year_d = prem_d[year]

    for k, v in record.drop(meta_cols, axis=1).to_dict(orient='records')[0].items():
        year_d[k] = v


with open('premise_explorer.json', 'w') as f:
    json.dump(container, f, indent=4, separators=(',', ': '))

# Ugly patch
r = Reconstruct(file_name='premise_explorer.json')
r.execute_file_parse()
