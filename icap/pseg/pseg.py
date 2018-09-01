import pandas as pd
import numpy as np

import os
import tempfile
from functools import reduce
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

    def get_util_nits(self):
        util_query = """
                select distinct
                        CAST((c.CPYearID - 1) as varchar) as Year,
                        RTrim(u.RateClass) as RateClass, u.Strata,
                        u.ParameterId, u.ParameterValue
                        --Exp(Sum(Log(u.ParameterValue))) as PFactor
                from UtilityParameterValue as u
                inner join CoincidentPeak as c
                        on c.CPID = u.CPID
                where
                        u.UtilityId = 'PSEG'
                        and u.ParameterValue > 0
                        """
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

    def get_sys_params_nits(self):
        """Get PSEG System parameters"""

        sys_query = """
                select
                        CAST(CPYearId-1 as varchar) as Year,
                        ParameterId, ParameterValue
                from SystemLoad
                where UtilityId = 'PSEG'"""

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
                        h.UsageDate as CPDate,
                        h.HourEnding as HourEnding,
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

        self.compute_nits()

        return meta_organize(self, tmp_2)

    def compute_nits(self):
        # Get values required
        records = self.records_.copy()
        util = self.get_util_nits()
        sys = self.get_sys_params_nits().copy()

        # Utility params
        gen_cap_scale = filter_rename_drop(util, 'GenCapScale')
        cap_pro_peak_ratio = filter_rename_drop(util, 'CapProfPeakRatio')
        loss_exp = filter_rename_drop(util, 'LossExpanFactor')

        util_params = pd.merge(
            pd.merge(gen_cap_scale, cap_pro_peak_ratio, on=['Year', 'RateClass', 'Strata'], how='left'),
            loss_exp, on=['Year', 'RateClass', 'Strata'], how='left')

        # System Load
        plcsf = filter_rename_drop(sys, 'PLCScaleFactor')
        cap_oblig = filter_rename_drop(sys, 'CapObligScale')
        fpr = filter_rename_drop(sys, 'ForecastPoolResv')
        final_rpm = filter_rename_drop(sys, 'FinalRPMzonal')
        sys_load = [plcsf, cap_oblig, fpr, final_rpm]

        # Merge system load
        sys_params = reduce(lambda left, right: pd.merge(left, right, on=['Year']), sys_load)

        # Initialize all records
        objs = list()
        for k, g in records.groupby(['PremiseId', 'Year', 'RateClass']):
            r = Record(*k)
            r.meter_type = 'INT'
            r.cp_df = g.sort_values(by='CPDate')[['Year', 'CPDate', 'HourEnding', 'Usage', 'RateClass']]
            r.cp_df['UsageAvg'] = r.cp_df['Usage'].mean()
            objs.append(r)

        # Join on system params for each object
        for obj in objs:
            obj.cp_df = pd.merge(obj.cp_df, util_params, on=['Year', 'RateClass'], how='left')
            obj.cp_df = pd.merge(obj.cp_df, sys_params, on=['Year'], how='left')
            
            obj.compute_plc()
            obj.string_builder()

        rw = RecordWriter(objs)
        rw.write()



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
                    --Year(m.StartDate) = 2015 and
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

        rec.to_csv('pseg_records.csv')
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


        self.compute_nits()

        # clean up meta-data
        return meta_organize(self, tmp_2)

    def compute_nits(self):
        # Get values required
        records = self.records_.copy ()
        util = self.get_util_nits()
        sys = self.get_sys_params_nits().copy()

        # Utility params
        gen_cap_scale = filter_rename_drop(util, 'GenCapScale')
        cap_pro_peak_ratio = filter_rename_drop(util, 'CapProfPeakRatio')
        loss_exp = filter_rename_drop(util, 'LossExpanFactor')

        util_params = pd.merge(
            pd.merge(gen_cap_scale, cap_pro_peak_ratio, on=['Year', 'RateClass', 'Strata'], how='left'),
            loss_exp, on=['Year', 'RateClass', 'Strata'], how='left')

        # System Load
        plcsf = filter_rename_drop(sys, 'PLCScaleFactor')
        cap_oblig = filter_rename_drop(sys, 'CapObligScale')
        fpr = filter_rename_drop(sys, 'ForecastPoolResv')
        final_rpm = filter_rename_drop(sys, 'FinalRPMzonal')
        sys_load = [plcsf, cap_oblig, fpr, final_rpm]

        # Merge system load
        sys_params = reduce(lambda left, right: pd.merge(left, right, on=['Year']), sys_load)

        # Initialize all records
        def dmd_usage_avg(grp):
            return (grp['SummerCycle'] * grp['Demand']).sum() / grp['SummerCycle'].sum() 


        # Initialize all records
        objs = list()
        for k, g in records.groupby(['PremiseId', 'Year', 'RateClass']):
            _g = g.copy()
            r = Record(*k)
            _g['UsageAvg'] = dmd_usage_avg(g)
            r.cp_df = _g.sort_values(by='StartDate')
            r.meter_type = 'DMD'
            objs.append(r)

        # Join on system params for each object
        for obj in objs:
            obj.cp_df = pd.merge(obj.cp_df, util_params, on=['Year', 'RateClass'], how='left')
            obj.cp_df = pd.merge(obj.cp_df, sys_params, on=['Year'], how='left')
            
            obj.compute_plc()
            obj.string_builder()

        rw = RecordWriter(objs)
        rw.write()


class PSEGRecipe:
    def __init__(self, conn=None, results=None):
        if (conn is None) or (results is None):
            raise Exception('conn=%s, results=%s)' % (conn, results,))
        self.conn = conn
        self.Results = results

    def run_all(self):
        intv = PSEGInterval(self.conn).compute_icap()
        dmd = PSEGDemand(self.conn).compute_icap()
        #con = PSEGConsumption(self.conn).compute_icap()

        #all_results = pd.concat([intv, dmd, con])
        all_results = pd.concat([intv, dmd])
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
    #return df

class RecordWriter:
    def __init__(self, records=None):
        assert(records is not None)
        self.records = records
        self.meter_type = records[0].meter_type
        
    def write(self, fp=None):
        if os.path.exists('/home/ubuntu/JustEnergy'):
            fp = '/home/ubuntu/JustEnergy/pseg_{}_nits.csv'.format(self.meter_type)
        elif fp is None:
            fp = os.path.join(tempfile.gettempdir(), 'pseg_{}_nits.csv'.format(self.meter_type))
        else:
            fp = os.path.join(os.path.abspath(__file__), 'pseg_{}_nits.csv'.format(self.meter_type))
            
        if self.meter_type == 'INT':    
            header = 'PREMISEID,RATECLASS,RUNDATE,'\
            'USAGE DATE 1, HOUR ENDING 1, CP 1 USAGE,'\
            'USAGE DATE 2, HOUR ENDING 2, CP 2 USAGE,'\
            'USAGE DATE 3, HOUR ENDING 3, CP 3 USAGE,'\
            'USAGE DATE 4, HOUR ENDING 4, CP 4 USAGE,'\
            'USAGE DATE 5, HOUR ENDING 5, CP 5 USAGE,'\
            'CAPOBLIGSCALE, FINALRPMZONAL, FORECASTPOOLRESV, GENCAPSCALE, LOSSEXPANFACTOR,'\
            'CAPPROFPEAKRATIO,METER TYPE,'\
            'CAPACITY PLANNNG YEAR,'\
            'PLC,'\
            'NITS'
        else:
            header = 'PREMISEID,RATECLASS,RUNDATE,'\
            'STARTDATE 1, ENDDATE 1, DEMAND 1 , SUMMER CYCLE, BILL CYCLE,'\
            'STARTDATE 2, ENDDATE 2, DEMAND 2 , SUMMER CYCLE, BILL CYCLE,'\
            'STARTDATE 3, ENDDATE 3, DEMAND 3 , SUMMER CYCLE, BILL CYCLE,'\
            'STARTDATE 4, ENDDATE 4, DEMAND 4 , SUMMER CYCLE, BILL CYCLE,'\
            'STARTDATE 5, ENDDATE 5, DEMAND 5 , SUMMER CYCLE, BILL CYCLE,'\
            'STARTDATE 6, ENDDATE 6, DEMAND 6 , SUMMER CYCLE, BILL CYCLE,'\
            'STARTDATE 7, ENDDATE 7, DEMAND 7 , SUMMER CYCLE, BILL CYCLE,'\
            'CAPOBLIGSCALE, FINALRPMZONAL, FORECASTPOOLRESV, GENCAPSCALE, LOSSEXPANFACTOR,'\
            'CAPPROFPEAKRATIO,METER TYPE,'\
            'CAPACITY PLANNNG YEAR,'\
            'PLC,'\
            'NITS'
            
        with open(fp, 'w') as fout:
            fout.write(header + os.linesep)
            
            for rec in self.records:
                fout.write(rec.string_record + os.linesep)

class Record:
    def __init__(self, premise_id=None, year=None, rateclass=None, strata=None):
        assert(premise_id is not None)
        assert(year is not None)
        assert(rateclass is not None)
        
        self.premise_id = premise_id
        self.year = year
        self.plcyear = str(int(year) + 1)
        self.rateclass = rateclass
        self.strata = strata
        self.rundate = datetime.now()
        
        self.cp_df = None
        self.plc = None
        self.nits = None
        self.meter_type = None
        
        self.string_record = None
        
    def compute_plc(self):
        assert(self.cp_df is not None)
        assert(self.meter_type is not None)
        
        factors = ['UsageAvg', 'LossExpanFactor', 'TransLoadScale']
        
        if self.meter_type == 'DMD':
            factors = factors + ['CapProfPeakRatio']
            
        
        
        # Add empty rows where missing
        # Set plc to NaN; required 5 values
        
        max_row = 5 if self.meter_type == 'INT' else 7
        if self.cp_df.shape[0] < max_row:
        
            # Get number of rows to add
            num_new_rows = max_row - self.cp_df.shape[0]

            # Empty series to append dataframe
            empty = pd.Series([np.NaN for _ in range(self.cp_df.shape[1])], index=self.cp_df.columns, name='empty')
            for r in range(num_new_rows):
                self.cp_df = self.cp_df.append(empty)
            if self.meter_type == 'INT':
                self.plc = np.nan
                return
            
            
        # Compute PLC
        factors = ['UsageAvg', 'CapObligScale', 'ForecastPoolResv', 'FinalRPMzonal', 'GenCapScale', 'LossExpanFactor']
        self.plc = self.cp_df[factors].product(axis=1).iloc[0]
        
    def compute_nits(self):
        assert(self.meter_type is not None)
        
        factors = ['UsageAvg', 'LossExpanFactor', 'TransLoadScale']
        if self.meter_type.upper() == 'DMD':
            factors = factors + ['CapProfPeakRatio']
        
        try:
            self.nits = self.cp_df[factors].product(axis=1)
        except KeyError as e:
            raise NotImplementedError
        
    def __repr__(self):
        return 'Record<premise={premise_id}, rateclass={rateclass}, strata={strata}, year={year}>'.format(**self.__dict__)
    
    def string_builder(self):
        if self.meter_type == 'INT':
            # Id, rateclass, rundate
            rec = '{premise_id},{rateclass},{rundate},'.format(**self.__dict__)

            # coincident peak date, hourending, usage
            for row in self.cp_df[['CPDate', 'HourEnding', 'Usage']].itertuples():
                _, cp, hour, usage = row
                rec += '{},{},{},'.format(cp, hour, usage)

            # Capacity Obligation Scale
            rec +=  str(self.cp_df['CapObligScale'].values[0])

            # Final RPM Zonal
            rec += ',' + str(self.cp_df['FinalRPMzonal'].values[0])

            # Forecast Pool Reserve
            rec += ',' + str(self.cp_df['ForecastPoolResv'].values[0])

            # Gen Cap Scale
            rec += ',' + str(self.cp_df['GenCapScale'].values[0])

            # Loss Expan Factor
            rec += ',' + str(self.cp_df['LossExpanFactor'].values[0])

            # Cap Prof Peak
            rec += ',' + str(self.cp_df['CapProfPeakRatio'].values[0])


            # Meter
            rec += ',' + self.meter_type

            # Year
            rec += ',{}'.format(self.plcyear)

            # PLC and NITS
            rec += ',{plc},{nits}'.format(**self.__dict__)
            self.string_record = rec
            return
        # Id, rateclass, rundate
        rec = '{premise_id},{rateclass},{rundate},'.format(**self.__dict__)

        # coincident peak date, hourending, usage
        for row in self.cp_df[['StartDate', 'EndDate', 'Demand', 'SummerCycle', 'BillCycle']].itertuples():
            _, sd, ed, dmd, sc, bc = row
            rec += '{},{},{},{},{},'.format(sd, ed, dmd, sc, bc)

        # Capacity Obligation Scale
        rec +=  str(self.cp_df['CapObligScale'].values[0])

        # Final RPM Zonal
        rec += ',' + str(self.cp_df['FinalRPMzonal'].values[0])

        # Forecast Pool Reserve
        rec += ',' + str(self.cp_df['ForecastPoolResv'].values[0])

        # Gen Cap Scale
        rec += ',' + str(self.cp_df['GenCapScale'].values[0])

        # Loss Expan Factor
        rec += ',' + str(self.cp_df['LossExpanFactor'].values[0])

        # Cap Prof Peak
        rec += ',' + str(self.cp_df['CapProfPeakRatio'].values[0])

        
        # Meter
        rec += ',' + self.meter_type

        # Year
        rec += ',{}'.format(self.plcyear)

        # PLC and NITS
        rec += ',{plc},{nits}'.format(**self.__dict__)
        self.string_record = rec

def filter_rename_drop(df, target):
    _filt = df[df.ParameterId == target].copy()
    return _filt.rename(columns={'ParameterValue': target}).drop(labels='ParameterId', axis=1)