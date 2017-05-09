import pandas as pd
import numpy as np
from datetime import datetime


class PPL:
    '''PPL has ONLY interval meters.
    '''

    def __init__(self, conn, meter_type='INT', premise=None):
        self.params = {'ISO': 'PJM',
                       'RunDate': datetime.now(),  # .__format__('%Y-%m-%d %H:%M'),
                       'Utility': 'PPL',
                       'MeterType': meter_type}

        # dynamic
        self.conn = conn
        self.meter_type = meter_type
        self.premise = premise

        # computed vars
        self.util_df_ = self.get_util_params()
        self.sys_df_ = self.get_sys_params()

    def get_util_params(self):
        """Get PPL Utility parameters"""

        util_query = """
                select
                        CAST(Year(StartDate)-1 as varchar) as Year,
                        cp.CPDate,
                        RTrim(RateClass) as RateClass,
                        RTrim(ParameterId) as ParameterId,
                        iif(ParameterId = 'RateClassLoss', 1. + ParameterValue/100., ParameterValue) as ParameterValue
                from [UtilityParameterValue] upv
                inner join [CoincidentPeak] cp
                        on cp.CPID = upv.CPID
                where
                        upv.UtilityId = 'PPL'
                order by RateClass, ParameterId"""

        # return DataFrame
        return pd.read_sql(util_query, self.conn)

    def get_sys_params(self):
        """Get PPL System parameters"""

        sys_query = """
                select
                        CAST(CPYearId-1 as varchar) as Year,
                        ParameterId, ParameterValue
                from SystemLoad
                where UtilityId = 'PPL'
                """

        return pd.read_sql(sys_query, self.conn)


class PPLInterval(PPL):
    def __init__(self, conn, premise=None, meter_type='INT'):
        PPL.__init__(self, conn, meter_type)

        self.premise = premise
        self.meter_type = meter_type
        self.records_ = self.get_records()

    def get_records(self):
        query = """
            select
                h.PremiseId,
                Cast(Year(h.UsageDate) as varchar) as Year,
                h.UsageDate,
                h.Usage,
                RTrim(p.RateClass) as RateClass,
                RTrim(p.Strata) as Strata
            from [HourlyUsage] h
            inner join [CoincidentPeak] cp
                on cp.UtilityId = h.UtilityId and
                cp.CPDate = h.UsageDate and
                cp.HourEnding = h.HourEnding
            inner join [Premise] p
                on p.UtilityId = h.UtilityId and
                p.PremiseId = h.PremiseId
            where
                h.UtilityId = 'PPL'
                {prem}"""

        if self.premise is not None:
            query = query.format(prem="and h.PremiseId = '%s'" % self.premise)
        else:
            query = query.format(prem="")

        return pd.read_sql(query, self.conn)

    def compute_icap(self):
        util = self.util_df_.copy()
        rec = self.records_.copy()
        sys = self.sys_df_.copy()


        # loss factor
        util = util[util['ParameterId'] == 'Loss Factor'].copy()


        # extract date-part (string) from parameterid
        sys['ParameterId'] = sys['ParameterId'].apply(lambda x: x.split()[0])
        sys.rename(columns={'ParameterId': 'CPDate'}, inplace=True)

        # handle missing data with np.NaN values
        grp = rec.groupby(['PremiseId', 'Year', 'RateClass'])['Usage'].agg(
            {'Count':len}).reset_index()

        bad_idx = grp[grp['Count'] != 5].index
        rec.set_value(bad_idx, 'Usage', np.nan)


        # merge( merge(records, sys), util)
        tmp = pd.merge(
            pd.merge(rec, sys,
                     left_on=['Year', 'UsageDate'],
                     right_on=['Year', 'CPDate'],
                     how='left'),
            util, on=['Year', 'RateClass'], how='left')

        # rename convienience
        tmp.rename(columns={'ParameterValue_x': 'ReconFactor',
                            'ParameterValue_y': 'RateClassLossFactor'},
                   inplace=True)

        # define the `apply` function
        def ppl_icap(g):
            return (g['Usage'] * g['ReconFactor'] * g['RateClassLossFactor']).mean()

        icap = tmp.groupby(['PremiseId', 'Year', 'RateClass', 'Strata']
                           ).apply(ppl_icap).reset_index()
        icap.rename(columns={0: 'ICap'}, inplace=True)

        icap['Strata'] = 'NULL'

        return meta_organize(self, icap)


class PPLRecipe:

    def __init__(self, conn=None, results=None):
        if (conn is None) or (results is None):
            raise Exception('conn=%s, results=%s)' % (conn, results,))
        self.conn = conn
        self.Results = results

    def run_all(self):
        intv = PPLInterval(self.conn).compute_icap()
        res = self.Results(self.conn, intv)
        return res


def meta_organize(obj_ref, df):
    """meta_organize updates the returned dataframe to include
    required information for database upload.

    The `params` dict is held in the PPL super class. Values
    are iterated over and assigned. Only the desired columns are
    returned.
    """
    keep = ['RunDate', 'ISO', 'Utility', 'PremiseId', 'Year',
            'RateClass', 'Strata', 'MeterType', 'ICap']

    # loop over params and update dataframe
    for k, v in obj_ref.params.items():
        df[k] = v

    # year must be adjusted to CPYear
    add_one = lambda yr: str(int(yr) + 1)
    df['Year'] = df['Year'].apply(add_one)
    return df[keep]
