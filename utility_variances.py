from icap.database.icapdatabase import ICapDatabase
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

fp = './icap/database/icapdatabase.json'
conn = ICapDatabase(fp).connect()

query = """select *, YEAR(StartDate) as Year 
	from [UtilityParameterValue]
	where UtilityId in ('PSEG', 'PECO')"""

df = pd.read_sql(query, conn)

stats = {'stats': ['mean', 'std']}

# Group by Utility, Year, Parameter, RateClass
norm = df.dropna(subset=['ParameterValue']).groupby(['UtilityID', 'RateClass', 'ParameterID'])['ParameterValue'].agg(stats)


print(norm)
norm.to_csv('utility_parameter_variances.csv')
#print(norm.sort_values(ascending=False))
#.std().sort_values(ascending=False).reset_index().to_csv('util_parm_variances.csv')
