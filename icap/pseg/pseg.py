import pandas as pd

class PSEG():

	def __init__(self, db_connection, premise=None):
		''' if no premise is passed, assume BATCH '''
		self.conn = db_connection
		self.premise = premise
		# computed vars
		self.util_df_ = self.get_util_params()
		self.sys_df_ = self.get_sys_params()
		self.records_ = self.get_records()

	def get_util_params(self):

		util_query = """
			select distinct
				CAST((c.CPYearID - 1) as varchar) as Year,
				RTrim(u.RateClass), u.Strata,
				u.ParameterId, u.ParameterValue
			from UtilityParameterValue as u
			inner join CoincidentPeak as c
				on c.CPID = u.CPID
			where u.UtilityId = 'PSEG'
				and u.ParameterId in ('GenCapScale','LossExpanFactor')
			order by u.RateClass, Year, ParameterId"""
	return pd.read_sql(util_query, self.conn)

	def get_sys_params(self):
		sys_query = """
			select 
				CAST(CPYearId-1 as varchar) as Year,
				Exp(Sum(Log(ParameterValue))) as PFactor
			from SystemLoad
			where UtilityId = 'PSEG'
				and ParameterId in ('CapObligScale', 'ForecastPoolResv', 'FinalRPMZonal')
			group by Cast(CPYearId-1 as varchar)"""

		return pd.read_sql(sys_query, self.conn)



	def get_records(self):
		record_query = """
			select 
				h.PremiseId,
				Cast(Year(h.UsageDate) as varchar) as Year,
				RTrim(p.RateClass),
				RTrim(p.Strata),
				h.Usage
			from HourlyUsage as h
			inner join CoincidentPeak as c
				on c.UtilityId = h.UtilityId
				and c.CPDate = h.UsageDate
				and c.HourEnding = h.HourEnding
			inner join Premise as p
				on p.UtilityId = h.UtilityId
				and p.PremiseId = h.PremiseId
			where h.UtilityId = 'PSEG'
				{prem}
			order by h.PremiseId, Year"""
		
		if self.premise:
			record_query = record_query.format(prem="and h.PremiseId = %s" % self.premise)
		else:
			record_query = record_query.format(prem="")


		return pd.read_sql(record_query, self.conn)


