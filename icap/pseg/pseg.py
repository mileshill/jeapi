import pandas as pd
import numpy as np
from datetime import datetime


class PSEG():
    '''SuperClass for meter types. Loads the system and
    utility parameters from DB
    '''

    def __init__(self, conn, meter_type=None):
        self.params = {'ISO': 'PJM',
                       'RunDate': datetime.now(),
                       'Utility': 'PSEG',
                       'MeterType': meter_type}

        # dynamic
        self.conn = conn
        self.meter_type = meter_type

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
                        and u.ParameterId in ('GenCapScale','LossExpanFactor'
                            {notinterval})
                        and u.ParameterValue > 0
                group by
                        CAST((c.CPYearID-1) as varchar),
                        RTrim(u.RateClass),
                        u.Strata"""

        # logic for correct utility factor selection
        if self.meter_type == 'CON':
            util_query = util_query.format(
                **{'notinterval': ",'CapProfPeakRatio'"})
        else:
            util_query = util_query.format(**{'notinterval': ''})

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
    """Computes the Interval Meter ICap value"""

    def __init__(self, conn, premise=None, meter_type='INT'):
        ''' if no premise is passed, assume BATCH '''
        PSEG.__init__(self, conn, meter_type)

        self.premise = premise
        self.meter_type = meter_type

        # computed vars
        self.records_ = self.get_records()

    def get_records(self):
        """Query database for interval records"""
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

        # if single premise, update query for that premise
        if self.premise:
            record_query = record_query.format(
                prem="and h.PremiseId = '%s'" % self.premise)
        # get batch records
        else:
            record_query = record_query.format(prem="")

        # return dataframe
        return pd.read_sql(record_query, self.conn)

    def compute_icap(self):
        """PSEG Interval ICAP:
        icap = avg(cp_usage) * util[rateclass, year] * sys[year]
        """
        # group records into multi-index and aggregate
        grp = self.records_.groupby(['PremiseId', 'Year', 'RateClass']
                                    )['Usage'].agg(
            {'Count': len, 'UsageAvg': np.mean}
        ).reset_index()

        # if Count != 5 then UsageAvg -> np.nan
        bad_idx = grp[grp['Count'] != 5].index
        grp.set_value(bad_idx, 'UsageAvg', np.nan)

        # merge with system and utility params; compute icap
        tmp = pd.merge(grp, self.util_df_, on=[
                       'Year', 'RateClass'], how='left')
        tmp_2 = pd.merge(tmp, self.sys_df_, on=['Year'], how='left')
        tmp_2['ICap'] = tmp_2['UsageAvg'] * \
            tmp_2['PFactor_x'] * tmp_2['PFactor_y']

        return meta_organize(self, tmp_2)


class PSEGConsumption(PSEG):
    def __init__(self, conn, premise=None, meter_type='CON'):
        PSEG.__init__(self, conn, meter_type)

        self.premise = premise
        self.meter_type = meter_type

        # computed vars
        self.records_ = self.get_records()

    def get_records(self):
        record_query = """
             select
                    m.PremiseId,
                    Cast(Year(m.StartDate) as varchar) as Year,
                    m.StartDate, m.EndDate, m.Usage,
                    RTrim(p.RateClass) as RateClass,
                    -- Select days in summer cycle
                    CASE
                            WHEN Month(m.StartDate) = 5 THEN Day(m.EndDate)
                            WHEN Month(m.EndDate) = 10 THEN 30 - Day(m.EndDate) + 1
                            ELSE DateDiff(d, m.StartDate, m.EndDate) + 1
                    END as SummerCycle,
                    -- Bill Cycle Length
                    DateDiff(d, m.StartDate, m.EndDate) + 1 as BillCycle
             from [MonthlyUsage] m
             inner join [Premise] p
                on p.UtilityId = m.UtilityId
                and p.PremiseId = m.PremiseId
             where
                    m.UtilityId = 'PSEG' and
                    m.Demand is Null and
                    (Month(m.StartDate) between 5 and 9) and
                    (Month(m.EndDate) between 6 and 10) and
                    m.PremiseId not in (
                        select distinct PremiseId
                        from HourlyUsage
                        where UtilityId = 'PSEG') and
                        RTrim(p.RateClass) in ('RS', 'RSH', 'RHS', 'RLM',' WH', 'WHS', 'HS')
                    {prem}"""

        if self.premise:
            record_query = record_query.format(
                prem="and m.PremiseId = '%s'" % self.premise)
        else:
            record_query = record_query.format(prem="")

        return pd.read_sql(record_query, self.conn)

    def compute_icap(self):
        """PSEG Consumption ICAP:
        icap = avg(cp_usage) * util[rateclass, year] * sys[year]
        """

        # BEGIN PREPROCESSING
        # NEW DATA: [SummerHours, DailyAvg, ActualSummerUsage]
        # get record set
        df = self.records_
        df['SummerHours'] = df['SummerCycle'] * 24.0
        df['DailyAvg'] = df['Usage'] / df['BillCycle']
        df['ActualSummerUsage'] = df['DailyAvg'] * df['SummerCycle']

        # Group and Aggregate
        # sum over [SummerHours, ActualSummerUsage]
        grp = df.groupby(['PremiseId', 'Year', 'RateClass'])
        tmp = grp[['SummerHours', 'ActualSummerUsage']].agg(np.sum)
        weighted_summer = pd.DataFrame(tmp['ActualSummerUsage'] / tmp['SummerHours']
                                       ).reset_index()
        weighted_summer.rename(columns={0: 'WeightedSummer'}, inplace=True)
        # END PREPROCESSING

        # BEGIN MERGE and COMPUTE
        # Merge utility and system parameters
        tmp_2 = pd.merge(weighted_summer, self.util_df_, on=['Year', 'RateClass'],
                         how='left')
        tmp_3 = pd.merge(tmp_2, self.sys_df_, on='Year', how='left')

        tmp_3['ICap'] = tmp_3['WeightedSummer'] * \
            tmp_3['PFactor_x'] * tmp_3['PFactor_y']
        # END MERGE and COMPUTE

        return meta_organize(self, tmp_3)


class PSEGDemand(PSEG):
    def __init__(self, conn, premise=None, meter_type='DMD'):
        PSEG.__init__(self, conn, meter_type)

        self.premise = premise
        self.meter_type = meter_type

        # computed vars
        self.records_ = self.get_records()

    def get_records(self):
        record_query = """
            select
                    m.PremiseId,
                    Cast(Year(m.StartDate) as varchar) as Year,
                    m.StartDate, m.EndDate,
                    m.Usage, m.Demand,
                    RTrim(p.RateClass) as RateClass,
                    -- Select days in summer cycle
                    CASE
                            WHEN Month(m.StartDate) = 5 THEN Day(m.EndDate)
                            WHEN Month(m.EndDate) = 10 THEN 30 - Day(m.StartDate)  + 1
                            ELSE DateDiff(d, m.StartDate, m.EndDate) + 1
                    END as SummerCycle,
                    -- Bill Cycle Length
                    DateDiff(d, m.StartDate, m.EndDate) + 1 as BillCycle
             from [MonthlyUsage] m
             inner join [Premise] p
                on p.UtilityId = m.UtilityId
                and p.PremiseId = m.PremiseId
             where
                    m.UtilityId = 'PSEG' and
                    Year(m.StartDate) = 2015 and
                    (m.Demand >= 0 or m.Demand is not Null) and
                    (Month(m.StartDate) between 5 and 9) and
                    (Month(m.EndDate) between 6 and 10) and
                    m.PremiseId not in (
                        select distinct PremiseId
                        from HourlyUsage
                        where UtilityId = 'PSEG')
                    {prem}"""

        if self.premise:
            record_query = record_query.format(
                prem="and m.PremiseId = '%s'" % self.premise)
        else:
            record_query = record_query.format(prem='')

        return pd.read_sql(record_query, self.conn)

    def compute_icap(self):
        """PSEG Demand ICAP"""
        rec = self.records_.copy()

        # compute summer demand; SummerCycle * Demand
        rec['SummerDemand'] = rec['SummerCycle'] * rec['Demand']

        # compute Generation Capacity Load; sum(SummerDemand) /
        # sum(SummerCycle)
        gen_cap_load = lambda grp: grp[
            'SummerDemand'].sum() / grp['SummerCycle'].sum()
        rec = rec.groupby(['PremiseId', 'Year', 'RateClass']
                          ).apply(gen_cap_load).reset_index()
        rec.rename(columns={0: 'GenCapLoad'}, inplace=True)

        # merge on utility and system factors
        tmp = pd.merge(rec, self.util_df_, on=[
                       'Year', 'RateClass'], how='left')
        tmp_2 = pd.merge(tmp, self.sys_df_, on=['Year'], how='left')

        # compute the icap
        tmp_2['ICap'] = tmp_2['GenCapLoad'] * \
            tmp_2['PFactor_x'] * tmp_2['PFactor_y']

        # clean up meta-data
        return meta_organize(self, tmp_2)


class PSEGRecipe:
    def __init__(self, conn=None, results=None):
        if (conn is None) or (results is None):
            raise Exception('conn=%s, results=%s)' % (conn, results,))
        self.conn = conn
        self.Results = results

    def run_all(self):
        intv = PSEGInterval(self.conn).compute_icap()
        dmd = PSEGDemand(self.conn).compute_icap()
        con = PSEGConsumption(self.conn).compute_icap()

        all_results = pd.concat([intv, dmd, con])
        res = self.Results(self.conn, all_results)
        return res


def meta_organize(obj_ref, df):
    """meta_organize updates the returned dataframe to include
    required information for database upload.

    The `params` dict is held in the PSEG super class. Values
    are iterated over and assigned. Only the desired columns are
    returned.
    """
    keep = ['RunDate', 'ISO', 'Utility', 'PremiseId', 'Year',
            'RateClass', 'Strata', 'MeterType', 'ICap']

    # loop over params and update dataframe
    for k, v in obj_ref.params.items():
        df[k] = v

    add_one = lambda yr: str(int(yr) + 1)
    df['Year'] = df['Year'].apply(add_one)
    return df[keep]
