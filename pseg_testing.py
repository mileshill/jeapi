#!/home/ubuntu/miniconda3/envs/predictions/bin/python3.5

from icap.database.icapdatabase import ICapDatabase
from icap.pseg.pseg import PSEGDemand

import pandas as pd


conn = ICapDatabase('/home/ubuntu/JustEnergy/icap/database/icapdatabase.json').connect()

p = PSEGDemand(conn)

rec = p.records_.copy()

rec['SummerDemand'] = rec['SummerCycle'] * rec['Demand']

gen_cap_load = lambda grp: grp['SummerDemand'].sum() / grp['SummerCycle'].sum()


rec = rec.groupby(['PremiseId', 'Year', 'RateClass']).apply(gen_cap_load).reset_index()
rec = rec.rename(columns={0: 'GenCapLoad'})


_tmp = pd.merge(rec, p.util_df_, on=['Year', 'RateClass'], how='left')

# Year members are type=__str__ and {2014, 2015}
print('Utility Years')
print(p.util_df_.Year.drop_duplicates())
print(type(p.util_df_.Year.ix[0]), '\n')


print('Record Years')
print(_tmp.shape)
print(_tmp.dropna().shape)
print(_tmp.Year.drop_duplicates())

print('\nSystem Years')
print(p.sys_df_.Year.drop_duplicates())


query = """
select 
	cast(CPYearId-1 as varchar) as Year,
	exp(sum(log(ParameterValue))) as PFactor
from [SystemLoad]
where
	UtilityId = 'PSEG' and
	ParameterId in ('CapObligScale', 'ForecastPoolResv', 'FinalRPMZonal')
group by
	cast(CPYearId-1 as varchar)
	
"""
print('SystemLoad query')
print(pd.read_sql(query, conn))

query = """select distinct CPYearId-1 as Year
from [SystemLoad]
where UtilityId = 'PSEG'""" 


print('#' * 10, 'System Load Years')
print(pd.read_sql(query, conn))
print('#' * 10)


from icap.pseg.pseg import PSEGConsumption
pc = PSEGConsumption(conn)
print(pc.sys_df_.Year.drop_duplicates())


print(pc.records_.head())
