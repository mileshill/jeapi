#!/home/ubuntu/miniconda3/envs/predictions/bin/python
from icap.database.icapdatabase import ICapDatabase
from icap.peco.peco import PECODemand

conn = ICapDatabase('./icap/database/icapdatabase.json').connect()

p = PECODemand(conn)




na_ratio = p.util_df_[p.util_df_['ParameterId'] == 'NARatio']

#print(
#	na_ratio.groupby(['Year', 'RateClass', 'Strata'])['ParameterValue'].mean()
#)


# print(p.records_['Year'].drop_duplicates())
#print(p.sys_df_[p.sys_df_['ParameterId'] == 'PLCScaleFactor']['Year'].drop_duplicates())

print(p.compute_icap())

#print(p.util_df_[p.util_df_['ParameterId'] == 'NARatio'])
#print(p.util_df_['ParameterId'].drop_duplicates())
print('All done.')
