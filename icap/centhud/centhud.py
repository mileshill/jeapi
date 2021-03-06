import pandas as pd
import numpy as np
from datetime import datetime


class CENTHUD():
    '''SuperClass for meter types. Loads the system and
    utility parameters from DB
    '''

    def __init__(self, conn, meter_type=None):
        self.params = {'ISO': 'NYISO',
                       'RunDate': datetime.now(),
                       'Utility': 'CENTHUD',
                       'MeterType': meter_type}

        # dynamic
        self.conn = conn
        self.meter_type = meter_type

        # computed vars
        self.util_df_ = self.get_util_params()
        self.sys_df_ = self.get_sys_params()

    def get_util_params(self):
        """Get CENTHUD Utility parameters"""

        util_query = """
            select distinct
                Cast(cp.CPYearID-1 as int) as Year,
                RTrim(upv.RateClass) as RateClass,
                RTrim(upv.Strata) as Strata,
                upv.ParameterValue as UtilityFactor
                --Exp(Sum(Log(upv.ParameterValue))) as UtilityFactor
            from [UtilityParameterValue] upv
            inner join [CoincidentPeak] cp
                on cp.CPID = upv.CPID
            where
                upv.UtilityId = 'CENTHUD' and
                upv.ParameterId = 'WeatherNormalFactor'
            """

        """group by
            cp.CPYearID-1,
            RTrim(upv.RateClass),
            RTrim(upv.Strata)"""

        return pd.read_sql(util_query, self.conn)

    def get_sys_params(self):
        """Get CENTHUD system parameters"""

        system_query = """
            select 
                Cast((CPYearId - 1) as int) as Year,
                (1.0 + ParameterValue) as VoltageLoss
            from [SystemLoad]
            where
                ParameterId = 'LossLossFACTOR'"""

        return pd.read_sql(system_query, self.conn)


class CENTHUDConsumption(CENTHUD):
    def __init__(self, conn, premise=None, meter_type='CON'):
        CENTHUD.__init__(self, conn, meter_type)

        self.premise = premise
        self.meter_type = meter_type

        # computed vars
        self.records_ = self.get_records()

    def get_records(self):
        record_query = """
               select distinct
                    m.PremiseId,
                    Year(m.EndDate) as Year,
                    RTrim(p.RateClass) as RateClass,
                    RTrim(p.Strata) as Strata,
                    lp.AvgHourlyLoad_kw as LoadProfile,
                    p.UsageFactor,
                    p.EffectiveStartDate, p.EffectiveStopDate,
                    cp.CPDate, cp.CPYearId
                from [MonthlyUsage] m
                inner join [CoincidentPeak] cp
                    on cp.UtilityId = m.UtilityId
                inner join [Premise] p
                    on p.UtilityId = m.UtilityId
                    and p.PremiseId = m.PremiseId
                    and Year(p.EffectiveStartDate) = cp.CPYearId 
                inner join [CENTHUD_Load_Profile] lp
                    on p.Strata = lp.Stratum
                    and Month(cp.CPDate) = Cast(lp.Month as int)
                    and (cp.HourEnding + 1) = Cast(lp.Hour as int)
                    and Year(p.EffectiveStartDate) = cp.CPYearId
                where
                    m.UtilityID = 'CENTHUD'
                    --and m.PremiseId = '1222106000'
                    and (m.Demand is NULL or m.Demand = 0)
                    and (cp.CPDate between m.StartDate and m.EndDate)
                    and lp.DayType = 'WKDAY'
                    {prem}"""

        if self.premise:
            record_query = record_query.format(
                prem="and m.PremiseId = '%s'" % self.premise)
        else:
            record_query = record_query.format(prem="")

        return pd.read_sql(record_query, self.conn)

    def compute_icap(self):
        """CENTHUD Consumption ICAP:
        """

        # Join records and utility factor
        tmp = pd.merge(self.records_, self.util_df_,
                       how='left',
                       on=['Year', 'Strata'])

        # Join on the system parameter
        tmp_2 = pd.merge(tmp, self.sys_df_,
                         how='left', on=['Year'])

        # Drop/Rename RateClass redundency; result of the joins
        tmp_2.drop('RateClass_y', axis=1, inplace=True)
        tmp_2.rename(columns={'RateClass_x':'RateClass'}, inplace=True)

        # Compute the ICap
        tmp_2['ICap'] = tmp_2['LoadProfile'] * \
            tmp_2['UtilityFactor'] * tmp_2['UsageFactor']
        tmp_2.drop_duplicates(inplace=True)

        return meta_organize(self, tmp_2)


class CENTHUDDemand(CENTHUD):
    def __init__(self, conn, premise=None, meter_type='DMD'):
        CENTHUD.__init__(self, conn, meter_type)

        self.premise = premise
        self.meter_type = meter_type

        # computed vars
        self.util = self.get_util_params()  # override parent
        self.load_profile = self.get_load_profile()
        self.records_ = self.get_records()

    def get_util_params(self):
        util_query = """
            select distinct
                Cast(cp.CPYearID-1 as int) as Year,
                RTrim(upv.RateClass) as RateClass,
                RTrim(upv.Strata) as Strata,
                --Exp(Sum(Log(upv.ParameterValue))) as UtilityFactor
                upv.ParameterValue as UtilityFactor
            from [UtilityParameterValue] upv
            inner join [CoincidentPeak] cp
                on cp.CPID = upv.CPID
            where
                upv.UtilityId = 'CENTHUD' and
                upv.ParameterId = 'WeatherNormalFactor'''"""

        return pd.read_sql(util_query, self.conn)

    def get_load_profile(self):
        load_profile_query = """
            select distinct
                RTrim(lp.Stratum) as Strata,
                lp.AVGHourlyLoad_kW as AVGHourlyLoad,
                Year(cp.CPDate) as Year
            from [CENTHUD_LOAD_PROFILE] lp
            inner join [CoincidentPeak] cp
                on cp.UtilityID = 'CENTHUD'
                and Month(cp.CPDate) = lp.Month
                and cp.HourEnding = lp.Hour
            where
                DayType = 'WKDAY'"""

        return pd.read_sql(load_profile_query, self.conn)

    def get_records(self):
        record_query = """
            select m.PremiseId,
                Cast(Year(m.EndDate) as int) as Year,
                RTrim(p.RateClass) as RateClass,
                RTrim(p.Strata) as Strata,
                Avg(m.Demand) as AVGDemand
            from [MonthlyUsage] m
            inner join [Premise] p
                on p.UtilityId = m.UtilityId
                and p.PremiseId = m.PremiseId
            where
                m.UtilityId = 'CENTHUD'
                and m.Demand is not NULL
                and Month(m.EndDate) in (6,7,8)
            group by
                m.PremiseId,
                Cast(Year(m.EndDate) as int),
                RTrim(p.RateClass),
                RTrim(p.Strata)
            having
                Count(m.EndDate) = 3
                    {prem}"""

        if self.premise:
            record_query = record_query.format(
                prem="and m.PremiseId = '%s'" % self.premise)
        else:
            record_query = record_query.format(prem='')

        return pd.read_sql(record_query, self.conn)

    def compute_icap(self):
        """CENTHUD Demand ICAP"""
        tmp = pd.merge(self.records_, self.util_df_,
                       how='left',
                       on=['Year', 'RateClass', 'Strata'])

        tmp = pd.merge(tmp, self.load_profile,
                       how='left',
                       on=['Strata', 'Year'])

        tmp['ICap'] = tmp['AVGDemand'] * tmp['UtilityFactor'] * \
            tmp['AVGHourlyLoad']

        return meta_organize(self, tmp)


class CENTHUDRecipe:
    def __init__(self, conn=None, results=None):
        if (conn is None) or (results is None):
            raise Exception('conn=%s, results=%s)' % (conn, results,))
        self.conn = conn
        self.Results = results

    def run_all(self):
        dmd = CENTHUDDemand(self.conn).compute_icap()
        con = CENTHUDConsumption(self.conn).compute_icap()

        all_results = pd.concat([dmd, con])
        res = self.Results(self.conn, all_results)
        return res


def meta_organize(obj_ref, df):
    """meta_organize updates the returned dataframe to include
    required information for database upload.

    The `params` dict is held in the CENTHUD super class. Values
    are iterated over and assigned. Only the desired columns are
    returned.
    """
    keep = ['RunDate', 'ISO', 'Utility', 'PremiseId', 'Year',
            'RateClass', 'Strata', 'MeterType', 'ICap']

    # loop over params and update dataframe
    for k, v in obj_ref.params.items():
        df[k] = v

    def adjust_icap_year(yr):
        return str(int(yr) + 1)

    df['Year'] = df['Year'].apply(adjust_icap_year)
    return df[keep]
