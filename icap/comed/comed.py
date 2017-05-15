import pandas as pd
import numpy as np
import pymssql
from datetime import datetime


class COMED():
    '''SuperClass for meter types. Loads the system and
    utility parameters from DB
    '''

    def __init__(self, conn, meter_type=None):
        self.params = {'ISO': 'PJM',
                       'RunDate': datetime.now(),
                       'Utility': 'COMED',
                       'MeterType': meter_type}

        # dynamic
        self.conn = conn
        self.meter_type = meter_type

        # static 
        self.normalized_peak_load = 20900000
        self.cust_delta = 1121264


class COMEDInterval(COMED):
    """Computes the Interval Meter ICap value"""

    def __init__(self, conn, premise=None, meter_type='INT'):
        ''' if no premise is passed, assume BATCH '''
        COMED.__init__(self, conn, meter_type)

        self.premise = premise
        self.meter_type = meter_type

        # computed vars
        self.cp_avg_peak_load = self.get_comed_cp_avg_peak_load(self.conn)
        self.pjm_cp_usage = self.get_pjm_cp_usage(self.conn)
        self.comed_cp_usage = self.get_comed_cp_usage(self.conn)
        self.load_drop_estimates = self.get_comed_load_drop_estimates(self.conn)
        self.utility_factors = self.get_comed_utility_factors(self.conn)

        # AcustCPL & AcustPL
        self.acustcpl = self.compute_acustcpl(self.conn)
        self.acustpl = self.compute_acustpl(self.conn)






    
    def get_comed_cp_avg_peak_load(self, conn: pymssql.Connection) -> pd.DataFrame:
        query = """
            select
                Cast(CPYearId-1 as int) Year,
                Avg(PJMZonalLoad) * 1000 as AvgCPZonalLoad
            from [CoincidentPeak]
            where UtilityId = 'COMED'
            group by
                Cast(CpYearId-1 as int)
            """
        df = pd.read_sql(query, conn).set_index('Year')
        df['Step4Diff'] = self.normalized_peak_load - df['AvgCPZonalLoad']
        return df

    @staticmethod
    def get_pjm_cp_usage(conn: pymssql.Connection, premise: str = None) -> pd.DataFrame:
        """Select all records from COMED Hourly
        for PJM coincident peak usage. Filter those values that
        do not possess 5 values per year
        """

        # query
        query = """
            select distinct
                h.PremiseId,
                iif(RTrim(p.RateClass)  is null, 'MILES', RTrim(p.RateClass)) as RateClass,
                RTrim(p.DSC) as DSC,
                CAST(cp.CPYearId -1 as INT) as Year, 
                cp.CPDate as CPDatePJM,
                cp.HourEnding as CPHourEnding,
                --cp.HourEnding-1 as ADJCPHourEndingPJM, 
                h.Usage as UsagePJM
            from [COMED_Premise] p
            inner join [HourlyUsage] h on
                p.PremiseId = h.PremiseId
            inner join [CoincidentPeak] cp on
                cp.UtilityId = h.UtilityId and
                cp.CPDate = h.UsageDate and
                cp.HourEnding = h.HourEnding 
            where
                h.UtilityId = 'COMED'
                {prem}
            order by
                h.PremiseId, RateClass, DSC, cp.CPDate
            """
        if premise is not None:
            pjm_cp_query = query.format(
                prem="and h.PremiseId = '%s'" % premise)
        else:
            pjm_cp_query = query.format(prem="", year="")

        # read query
        df = pd.read_sql(pjm_cp_query, conn)

        # group by premise
        # create filter for len(usage) != 5
        grp = df.groupby(['PremiseId', 'Year'])['UsagePJM'].agg(
            {'CountPJM': len, 'MeanPJM': np.mean})
        grp.reset_index(inplace=True)

        # set `Mean` = np.NaN if `Count` != 5
        missing_data_index = grp[grp['CountPJM'] != 5.0].index
        grp = grp.set_value(missing_data_index, 'MeanPJM', np.nan)

        return pd.merge(df, grp, how='left',
                        on=['PremiseId', 'Year'])

    # def get_records(self):
    #     """Query database for interval records"""
    #     record_query = """
    #             select distinct
    #                     h.PremiseId,
    #                     Cast(Year(h.UsageDate) as varchar) as Year,
    #                     RTrim(p.RateClass) as RateClass,
    #                     RTrim(p.Strata) as Strata,
    #                     h.Usage
    #             from HourlyUsage as h
    #             inner join CoincidentPeak as c
    #                     on c.UtilityId = h.UtilityId
    #                     and c.CPDate = h.UsageDate
    #                     and c.HourEnding = h.HourEnding
    #             inner join Premise as p
    #                     on p.UtilityId = h.UtilityId
    #                     and p.PremiseId = h.PremiseId
    #             where h.UtilityId = 'COMED'
    #                     {prem}
    #             order by h.PremiseId, Year"""

    #     # if single premise, update query for that premise
    #     if self.premise:
    #         record_query = record_query.format(
    #             prem="and h.PremiseId = '%s'" % self.premise)
    #     # get batch records
    #     else:
    #         record_query = record_query.format(prem="")

    #     # return dataframe
    #     return pd.read_sql(record_query, self.conn)

    @staticmethod
    def get_comed_cp_usage(conn: pymssql.Connection, premise: str = None) -> pd.DataFrame:
        """Select all records from COMED Hourly
        for COMED coincident peak usage. Filter those values that
        do not possess 5 values per year
        """

        # query
        query = """
            select distinct
                h.PremiseId,
                iif(RTrim(p.RateClass)  is null, 'MILES', RTrim(p.RateClass)) as RateClass,
                RTrim(p.DSC) as DSC,
                CAST(cp.CPYearId -1 as INT) as Year, 
                cp.CPDate as CPDateCOMED, 
                cp.HourEnding as CPHourEndingCOMED, 
                h.Usage as UsageCOMED,
                ZonalLoad
            from [HourlyUsage] h
            inner join [COMED_CoincidentPeak] cp on
                cp.UtilityId = h.UtilityId and
                cp.CPDate = h.UsageDate and
                cp.HourEnding = h.HourEnding
            inner join [COMED_Premise] p on
                p.PremiseId = h.PremiseId
            where
                h.UtilityId = 'COMED'
                {prem}
            order by
                h.PremiseId, RateClass, DSC, cp.CPDate
            """
        # format query for single premise
        if premise is not None:
            pjm_cp_query = query.format(
                prem="and h.PremiseId = '%s'" % premise)
        else:
            pjm_cp_query = query.format(prem="")

        # read query
        df = pd.read_sql(pjm_cp_query, conn)

        # group by premise
        # create filter for len(usage) != 5
        #df.replace(to_replace=None, value="Miles", inplace=True)

        grp = df.groupby(['PremiseId', 'Year', 'RateClass'])['UsageCOMED'].agg(
            {'CountCOMED': len, 'MeanCOMED': np.mean})
        grp.reset_index(inplace=True)

        # set `Mean` = np.NaN if `Count` != 5
        missing_data_index = grp[grp['CountCOMED'] != 5.0].index
        grp = grp.set_value(missing_data_index, 'MeanCOMED', np.nan)

        return pd.merge(df, grp, how='left',
                        on=['PremiseId', 'Year', 'RateClass'])

    @staticmethod
    def get_comed_load_drop_estimates(conn: pymssql.Connection) -> pd.DataFrame:
        query = """
            select
                Cast(CPYearID -1 as INT) as Year,
                ParameterId,
                (1.0 + ParameterValue/100.0) as LoadDrop
            from [SystemLoad]
            where
                UtilityId = 'COMED' --and
                --ParameterId = 'UFT'
        """

        # return pd.read_sql(query, conn).set_index('Year')
        df = pd.read_sql(query, conn)
        return pd.pivot_table(df, index='Year', columns='ParameterId', values='LoadDrop')

    @staticmethod
    def get_comed_utility_factors(conn: pymssql.Connection) -> pd.DataFrame:
        query = """
            select
                Year(StartDate) as Year,
                RTrim(RateClass) as DSC,
                ParameterId, ParameterValue
            from [UtilityParameterValue]
            where
                UtilityId = 'COMED'
                
        """

        df = pd.read_sql(query, conn)

        piv = pd.pivot_table(
            df, index=['Year', 'DSC'], columns='ParameterId', values='ParameterValue')

        return piv.reset_index(level=1)

    
    def compute_acustcpl(self, conn: pymssql.Connection, premise: str=None) -> pd.DataFrame:
        # mean_pjm = get_pjm_cp_usage(conn, premise=premise)[['PremiseId', 'Year', 'DSC', 'MeanPJM']] \
        #     .drop_duplicates() \
        #     .reset_index()


        mean_pjm = self.pjm_cp_usage[['PremiseId', 'Year', 'DSC', 'MeanPJM']] \
            .drop_duplicates() \
            .reset_index()





        # System and util factors index on Year
        util = self.utility_factors
        #util = get_comed_utility_factors(conn)
        #sys = pd.DataFrame(get_comed_load_drop_estimates(conn)['UFC'])
        sys = pd.DataFrame(self.load_drop_estimates['UFC'])

        # Join `util` and `sys` on Year index
        df = pd.merge(util, sys, left_index=True,
                      right_index=True).reset_index()

        # Join mean usage values with utility/system factors
        df = pd.merge(mean_pjm, df, on=['Year', 'DSC']).drop('index', axis=1)

        # Compute the AcustPL value
        df['AcustCPL'] = df['MeanPJM'] * df['DistLossFactor'] * \
            df['TransLossFactor'] * df['UFC']  # df['LoadDrop']
        df.set_index(['PremiseId', 'Year'], inplace=True)
        return df

    
    def compute_acustpl(self, conn: pymssql.Connection, premise: str=None) -> pd.DataFrame:
        # ComedCP unique mean musage values per year; includes np.NaN
        # mean_comed = get_comed_cp_usage(conn, premise=premise)[['PremiseId', 'Year', 'DSC', 'MeanCOMED']] \
        #     .drop_duplicates() \
        #     .reset_index()


        mean_comed = self.comed_cp_usage[['PremiseId', 'Year', 'DSC', 'MeanCOMED']] \
            .drop_duplicates() \
            .reset_index()



        # System and util factors index on Year
        util = self.utility_factors
        #util = get_comed_utility_factors(conn)
        #sys = pd.DataFrame(get_comed_load_drop_estimates(conn)['UFC'])
        sys = pd.DataFrame(self.load_drop_estimates['UFT'])



        # Join `util` and `sys` on Year index
        df = pd.merge(util, sys, left_index=True,
                      right_index=True).reset_index()

        # Join mean usage values with utility/system factors
        df = pd.merge(mean_comed, df, on=['Year', 'DSC']).drop('index', axis=1)

        # Compute the AcustPL value
        df['AcustPL'] = df['MeanCOMED'] * df['DistLossFactor'] * \
            df['TransLossFactor'] * df['UFT']  # df['LoadDrop']
        df.set_index(['PremiseId', 'Year'], inplace=True)
        return df

    
    def step6(self, r: pd.Series) -> np.float32:
        if r.AcustCPL >= r.AcustPL:
            return r.AcustCPL

        # *Step4Diff  #+ r.AcustCPL
        return ((r.AcustPL - r.AcustCPL) / self.cust_delta)

    @staticmethod
    def icap(r: pd.Series) -> np.float32:
        if r.AcustCPL == r.Step6:
            return r.AcustCPL
        return r.AcustCPL + r.Step7

    
    def compute_icap(self):

        peak_loads = pd.merge(pd.DataFrame(self.acustcpl), pd.DataFrame(self.acustpl),
            left_index=True, right_index=True)

        df = pd.merge(peak_loads, self.cp_avg_peak_load,
            left_index=True, right_index=True)
        
        df['Step6'] = df.apply(self.step6, axis=1)
        df['Step7'] = df['Step6'] * df['Step4Diff']
    
        df['ICap'] = df.apply(self.icap, axis=1)
        df['Strata'] = df['DSC_x']

        df.rename(columns={'DSC_x':'RateClass'}, inplace=True)
        df.reset_index(inplace=True)

        return meta_organize(self, df)



class COMEDRecipe:
    def __init__(self, conn=None, results=None):
        if (conn is None) or (results is None):
            raise Exception('conn=%s, results=%s)' % (conn, results,))
        self.conn = conn
        self.Results = results

    def run_all(self):
        intv = COMEDInterval(self.conn).compute_icap()
       
        #all_results = pd.concat([intv])
        res = self.Results(self.conn, intv)
        return res


def meta_organize(obj_ref, df):
    """meta_organize updates the returned dataframe to include
    required information for database upload.

    The `params` dict is held in the COMED super class. Values
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
