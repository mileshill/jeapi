import pandas as pd
import numpy as np


class PSEGResults():
    def __init__(self):
        pass

    def results(self):
        return self.icap_




class PSEG():
    '''SuperClass for meter types. Loads the system and
    utility parameters from DB
    '''
    def __init__(self, conn):
        self.conn = conn

        # computed vars
        self.util_df_ = self.get_util_params()
        self.sys_df_ = self.get_sys_params()

    def get_util_params(self):
        """Get PSEG Utility parameters"""

        util_query = """
                select distinct
                        CAST((c.CPYearID - 1) as varchar) as Year,
                        RTrim(u.RateClass) as RateClass, u.Strata,
                        Exp(Sum(Log(u.ParameterValue))) as PFactor
                from UtilityParameterValue as u
                inner join CoincidentPeak as c
                        on c.CPID = u.CPID
                where
                        u.UtilityId = 'PSEG'
                        and u.ParameterId in ('GenCapScale','LossExpanFactor')
                        and u.ParameterValue > 0
                group by
                        CAST((c.CPYearID-1) as varchar),
                        RTrim(u.RateClass),
                        u.Strata"""
        return pd.read_sql(util_query, self.conn)

    def get_sys_params(self):
        """Get PSEG System parameters"""

        sys_query = """
                select
                        CAST(CPYearId-1 as varchar) as Year,
                        Exp(Sum(Log(ParameterValue))) as PFactor
                from SystemLoad
                where UtilityId = 'PSEG'
                        and ParameterId in ('CapObligScale', 'ForecastPoolResv', 'FinalRPMZonal')
                group by Cast(CPYearId-1 as varchar)"""

        return pd.read_sql(sys_query, self.conn)




class PSEGInterval(PSEG):

    def __init__(self, conn, premise=None, meter_type='INT'):
        ''' if no premise is passed, assume BATCH '''
        PSEG.__init__(self, conn)
        self.premise = premise
        self.meter_type = meter_type

        # computed vars
        self.records_ = self.get_records()



    def get_records(self):
        record_query = """
                select
                        h.PremiseId,
                        Cast(Year(h.UsageDate) as varchar) as Year,
                        RTrim(p.RateClass) as RateClass,
                        RTrim(p.Strata) as Strata,
                        iif(h.Usage < 0, 0, h.Usage) as Usage
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
            record_query = record_query.format(prem="and h.PremiseId = '%s'" % self.premise)
        else:
            record_query = record_query.format(prem="")


        return pd.read_sql(record_query, self.conn)


    def compute_icap(self):
        """PSEG Interval ICAP:
        icap = avg(cp_usage) * util[rateclass, year] * sys[year]
        """
        res_obj = PSEGResults()
        results = []
        for key, grp in self.records_.groupby(['PremiseId', 'Year']):
            prem, year = key
            rate_class = grp.RateClass.values[0]
            count = len(grp)

            record = {'Premise': prem, 'Year': year, 'Count': count,
                      'RateClass':rate_class,'ICap':np.nan}
            if len(grp) != 5:
                results.append(record)
                continue

            sys = self.sys_df_[self.sys_df_.Year == year].PFactor.values[0]
            util = self.util_df_[(self.util_df_.Year == year) &
                              (self.util_df_.RateClass == rate_class)
                             ].PFactor.values.prod()

            avg_usage = np.mean(grp.Usage)


            record['ICap'] = avg_usage * util * sys
            results.append(record)

        res_obj.icap_ = pd.DataFrame(results)

        return res_obj

