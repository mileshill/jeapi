import pandas as pd
import numpy as np
from datetime import datetime


class PECO():
    '''SuperClass for meter types. Loads the system and
    utility parameters from DB
    '''

    def __init__(self, conn, meter_type=None):
        self.params = {'ISO': 'PJM',
                       'RunDate': datetime.now(),
                       'Utility': 'PECO',
                       'MeterType': meter_type}

        # dynamic
        self.conn = conn
        self.meter_type = meter_type

        # computed vars
        self.util_df_ = self.get_util_params()
        self.sys_df_ = self.get_sys_params()
        self.loadshape_df_ = self.get_load_shape()

    def get_util_params(self):
        """Get PECO Utility parameters"""

        util_query = """
                select
                        CAST(Year(StartDate)-1 as varchar) as Year,
                        cp.CPDate,
                        RTrim(RateClass) as RateClass,
                        RTrim(Strata) as Strata,
                        RTrim(ParameterId) as ParameterId,
                        iif(ParameterId = 'RateClassLoss', 1. + ParameterValue/100., ParameterValue) as ParameterValue
                from [UtilityParameterValue] upv
                inner join [CoincidentPeak] cp
                        on cp.CPID = upv.CPID
                where
                        upv.UtilityId = 'PECO'
                order by RateClass, Strata, ParameterId"""

        # logic for correct utility factor selection
        if self.meter_type == 'CON':
            util_query = util_query.format(
                **{'notinterval': ",'CapProfPeakRatio'"})
        else:
            util_query = util_query.format(**{'notinterval': ''})

        # return dataframe
        return pd.read_sql(util_query, self.conn)

    def get_sys_params(self):
        """Get PECO System parameters"""

        sys_query = """
                select
                        CAST(CPYearId-1 as varchar) as Year,
                        ParameterId, ParameterValue
                from SystemLoad
                where UtilityId = 'PECO'
                """

        return pd.read_sql(sys_query, self.conn)

    def get_load_shape(self):
        load_shape_query = """
            select distinct
                    RTrim(RateClass) as RateClass,
                    RTrim(Strata) as Strata,
                    (ConstantCoefficient + 99 * LinearCoefficient) as LoadShape
            from [SeasonLoadShape]
            where
                    DayType = 'WEEKDAY' and
                    Season = 'Summer' and
                    HourEnding = 17 and
                    Segment = 4 and
                    UpBandEffTemp = 200
                """
        return pd.read_sql(load_shape_query, self.conn)


class PECOInterval(PECO):
    """Computes the Interval Meter ICap value"""

    def __init__(self, conn, premise=None, meter_type='INT'):
        ''' if no premise is passed, assume BATCH '''
        PECO.__init__(self, conn, meter_type)

        self.premise = premise
        self.meter_type = meter_type

        # computed vars
        self.records_ = self.get_records()

    def get_records(self):
        """Query database for interval records"""
        record_query = """
               select
                    h.PremiseId, h.UsageDate, h.HourEnding, h.Usage,
                    Cast(Year(h.UsageDate) as varchar) as Year,
                    RTrim(p.RateClass) as RateClass,
                    RTrim(p.Strata) as Strata
                from [HourlyUsage] h
                -- extract coincident peak usage
                inner join [CoincidentPeak] cp
                        on cp.UtilityId = h.UtilityId
                        and cp.CPDate = h.UsageDate
                        and cp.HourEnding = h.HourEnding
                -- premise rateclass and strata
                inner join [Premise] p
                        on p.UtilityId = h.UtilityId
                        and p.PremiseId = h.PremiseId
                -- only return PECO information
                where
                        h.UtilityId = 'PECO' and
                        cp.CPDate between p.EffectiveStartDate and p.EffectiveStopDate
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
        """PECO Interval ICAP:
        icap = avg(cp_usage) * util[rateclass, year] * sys[year]
        """
        # copy records and utility values
        rec = self.records_.copy()
        util = self.util_df_.copy()

        # BEGIN PREPROCESSING
        # 1. Obtain index values for required utility params
        # 2. Merge records with correct NCRatio factor

        # 1. INDEX
        # index values for RateClassLoss and NCRatio
        rc_idx = util[util['ParameterId'] == 'RateClassLoss'].index
        nc_idx = util[util['ParameterId'] == 'NCRatio'].index

        # 2. MERGE
        # (record usage * ncratio) is unique to RateClass, Statra, Year, Date
        rec = pd.merge(rec, util.ix[nc_idx], how='left',
                       left_on=['Year', 'UsageDate', 'RateClass', 'Strata'],
                       right_on=['Year', 'CPDate', 'RateClass', 'Strata'])
        rec.rename(columns={'ParameterValue': 'NCRatio'}, inplace=True)
        # END PREPROCESSING

        # BEGIN WEATHER CORRECTION FACTOR
        # wcf_i = usage_i * ncratio_i
        # wcf = mean(wcf_i)
        rec['WCF'] = rec['Usage'] * rec['NCRatio']
        grp = rec.groupby(['PremiseId', 'Year', 'RateClass', 'Strata']
                          )['WCF'].agg({'Count': len, 'WCF': np.mean}).reset_index()

        # Convert to np.nan where insufficent/bad records
        # Condition 1: if count != 5 then wcf -> np.nan
        # Condition 2: if wcf == 0 then wcf -> np.nan
        bad_idx = grp[(grp['Count'] != 5) | (grp['WCF'] == 0.0)].index
        grp.set_value(bad_idx, 'WCF', np.nan)
        # END WEATHER CORRECTION FACTOR

        # BEGIN RATECLASSLOSS
        # Duplicates exists for the RateClassLoss. This is a scalar factor.
        # Remove duplicates and unnecessary columns
        keep = ['Year', 'RateClass', 'Strata', 'ParameterValue']
        rate_class = util.ix[rc_idx][keep].drop_duplicates()

        # Merge rate_class with grouped records
        tmp = pd.merge(grp, rate_class,
                       on=['Year', 'RateClass', 'Strata'], how='left')
        # END RATECLASSLOSS

        # ICAP
        # icap = wcf * rate_class_loss_factor ; (ParameterValue)
        tmp['ICap'] = tmp['WCF'] * tmp['ParameterValue']

        return meta_organize(self, tmp)


class PECOConsumption(PECO):
    def __init__(self, conn, premise=None, meter_type='CON'):
        PECO.__init__(self, conn, meter_type)

        self.premise = premise
        self.meter_type = meter_type

        # computed vars
        self.records_ = self.get_records()

    def get_records(self):
        record_query = """
           select distinct
                    m.PremiseId,
                    Cast(Year(m.StartDate) as varchar) as Year,
                    RTrim(p.RateClass) as RateClass,
                    RTrim(p.Strata) as Strata
            from [MonthlyUsage] m
            inner join [Premise] p
                    on p.UtilityId = m.UtilityId
                    and p.PremiseId = m.PremiseId
            where
                    RTrim(p.RateClass) in ('R', 'RH', 'OP') and
                    m.Demand is Null and
                    m.UtilityId = 'PECO' and
                    m.PremiseId not in (
                        select distinct PremiseId
                        from HourlyUsage
                        where UtilityId = 'PECO') and
                    (m.StartDate between p.EffectiveStartDate and p.EffectiveStopDate)
                   {prem}"""

        if self.premise:
            record_query = record_query.format(
                prem="and m.PremiseId = '%s'" % self.premise)
        else:
            record_query = record_query.format(prem="")

        return pd.read_sql(record_query, self.conn)

    def compute_icap(self):
        """PECO Consumption ICAP:
        icap = avg(cp_usage) * util[rateclass, year] * sys[year]
        """

        rec = self.records_.copy()
        util = self.util_df_.copy()
        sys = self.sys_df_.copy()
        load = self.loadshape_df_.copy()

        # BEGIN PREPROCESSING
        # 1. Compute utility factor
        # 2. Compute PLC factor

        # 1. Compute utility factor
        # Extract the required parameters
        cleaned_utl = util[(util['ParameterId'] == 'RateClassLoss') |
                           (util['ParameterId'] == 'StrataSummerScale')
                           ].drop_duplicates()
        # Group by [Year, RateClass, Strata]; compute the count and product of
        # ParameterValue for each group
        utl_grp = cleaned_utl.groupby(['Year', 'RateClass', 'Strata']
                                      )['ParameterValue'].agg(
            {'Count': len, 'UtilFactor': np.prod}).reset_index()

        # if count != 2 then UtilFactor -> np.nan
        bad_idx = utl_grp[utl_grp['Count'] != 2].index
        utl_grp.set_value(bad_idx, 'UtilFactor', np.nan)

        # 2. Compute PLC factor
        # collect plc index; extract values; rename ParameterValue -> PLC
        plc_idx = sys[sys['ParameterId'] == 'PLCScaleFactor'].index
        sys = sys.ix[plc_idx]
        sys.rename(columns={'ParameterValue': 'PLC'}, inplace=True)
        # END PREPROCESSING

        # BEGIN MERGE
        # 1. tmp = records & util
        # 2. tmp_2 = records & util & system
        # 3. temp_3 = records & util & system & loadshape
        tmp = pd.merge(rec, utl_grp,
                       on=['Year', 'RateClass', 'Strata'], how='left')

        tmp_2 = pd.merge(tmp, sys, on=['Year'], how='left')
        tmp_3 = pd.merge(tmp_2, load, on=['RateClass', 'Strata'], how='left')
        # END MERGE

        # ICAP
        tmp_3['ICap'] = tmp_3['UtilFactor'] * tmp_3['PLC'] * tmp_3['LoadShape']
        return meta_organize(self, tmp_3)


class PECODemand(PECO):
    def __init__(self, conn, premise=None, meter_type='DMD'):
        PECO.__init__(self, conn, meter_type)

        self.premise = premise
        self.meter_type = meter_type

        # computed vars
        self.records_ = self.get_records()

    def get_records(self):
        record_query = """
                select distinct
                        m.PremiseId,
                        Cast(Year(m.StartDate) as varchar) as Year,
                        m.StartDate,
                        RTrim(p.RateClass) as RateClass,
                        RTrim(p.Strata) as Strata,
                        m.Demand
                from [MonthlyUsage] m
                inner join [Premise] p
                        on p.UtilityId = m.UtilityId
                        and p.PremiseId = m.PremiseId
                where
                        (m.Demand is not Null or m.demand >= 0) and
                        m.UtilityId = 'PECO' and
                        Month(m.EndDate) in (6, 7, 8, 9) and
                        m.PremiseId not in (
                            select distinct PremiseId
                            from HourlyUsage
                            where UtilityId = 'PECO') and
                        (m.StartDate between p.EffectiveStartDate and p.EffectiveStopDate)
                               {prem}"""

        if self.premise:
            record_query = record_query.format(
                prem="and m.PremiseId = '%s'" % self.premise)
        else:
            record_query = record_query.format(prem='')

        return pd.read_sql(record_query, self.conn)

    def compute_icap(self):
        """PECO Demand ICAP"""
        rec = self.records_.copy()
        util = self.util_df_.copy()
        sys = self.sys_df_.copy()

        # BEGIN PREPROCESSING
        # 1. Obtain indicies for rateclassloss and naratio
        # 2. Compute avg demand per premise, year
        # 3. Compute avg naratio per year, rateclass, strata
        # 4. Prep the plc and rclf factors

        # 1.
        # need indices for RateClassLoss and NARatio
        rc_idx = util[util['ParameterId'] == 'RateClassLoss'].index
        na_idx = util[util['ParameterId'] == 'NARatio'].index

        # 2.
        redrec = rec.groupby(['PremiseId', 'Year', 'RateClass', 'Strata']
                             )['Demand'].agg({'Count': len, 'DmdAvg': np.mean}).reset_index()

        # set insufficent records to NaN
        # if count != 4 or dmdavg == 0 then dmdavg -> np.nan
        bad_rec_idx = redrec[(redrec['Count'] != 4) |
                             (redrec['DmdAvg'] == 0.0)].index
        redrec.set_value(bad_rec_idx, 'DmdAvg', np.nan)

        # 3.
        nared = util.ix[na_idx].groupby(['Year', 'RateClass', 'Strata']
                                        )['ParameterValue'].agg(
            {'Count': len, 'NAAvg': np.mean}
        ).reset_index()
        # set insufficent records to NaN
        # if count != 5 or naavg == 0 then naavg -> np.nan
        bad_na_idx = nared[(nared['Count'] != 5.0) |
                           (nared['NAAvg'] == 0.0)].index
        nared.set_value(bad_na_idx, 'NAAvg', np.nan)

        # 4.
        # plc factor
        # ['Year', 'PLC']
        plc_idx = sys[sys['ParameterId'] == 'PLCScaleFactor'].index
        plc = sys.ix[plc_idx][['Year', 'ParameterValue']]
        plc.rename(columns={'ParameterValue': 'PLC'}, inplace=True)

        # rclf factor
        # ['Year', 'RateClass', 'Strata', 'RCLF']
        rclf = util.ix[rc_idx].drop(axis=1, labels=['CPDate', 'ParameterId']
                                    ).rename(columns={'ParameterValue': 'RCLF'})
        # END PREPROCESSING

        # BEGIN MERGE
        tmp = pd.merge(redrec, nared,
                       on=['Year', 'RateClass', 'Strata'], how='left')

        tmp_2 = pd.merge(tmp, rclf,
                         on=['Year', 'RateClass', 'Strata'], how='left')

        tmp_3 = pd.merge(tmp_2, plc, on='Year', how='left')
        # END MERGE

        # ICAP
        tmp_3['ICap'] = tmp_3['DmdAvg'] * \
            tmp_3['NAAvg'] * tmp_3['RCLF'] * tmp_3['PLC']

        return meta_organize(self, tmp_3)


class PECORecipe:
    """Runs all meters types"""

    def __init__(self, conn=None, results=None):
        if (conn is None):
            raise Exception('conn=%s, results=%s' % (conn, results,))
        self.conn = conn
        self.Results = results

    def run_all(self):
        intv = PECOInterval(self.conn).compute_icap()
        dmd = PECODemand(self.conn).compute_icap()
        con = PECOConsumption(self.conn).compute_icap()

        all_results = pd.concat([intv, dmd, con])
        res = self.Results(self.conn, all_results)
        return res


def meta_organize(obj_ref, df):
    """meta_organize updates the returned dataframe to include
    required information for database upload.

    The `params` dict is held in the PECO super class. Values
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
