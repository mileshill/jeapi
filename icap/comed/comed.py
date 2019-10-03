import pandas as pd
import numpy as np
import pymssql
from datetime import datetime
import os
import tempfile

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
        # self.normalized_peak_load = 20900000
        # self.cust_delta = 1121264

        self.normalized_peak_loads = self.get_normalized_peak_loads()
        self.cust_delta = self.get_cust_delta()
        self.pjm_records = self.get_pjm_cp_records()
        self.comed_records = self.get_comed_cp_records()
        self.dsc_records = self.get_delivery_service_class()

    def get_pjm_cp_records(self):
        query = """
            select distinct
                h.PremiseId premise_id, 
                (cp.CPYearID - 1) year,cp.CPDate pjm_cp_date,
                AVG(h.Usage) pjm_avg_usage
            from HourlyUsage h
            inner join CoincidentPeak cp
                on cp.CPDate = h.UsageDate AND
                cp.HourEnding = h.HourEnding
            where h.UtilityId = 'COMED'
            group by 
                h.PremiseId, (cp.CPYearID - 1), cp.CPDate
            order by h.PremiseId, cp.CPDate
        """
        return pd.read_sql(query, self.conn)

    def get_comed_cp_records(self):
        query = """
        select distinct
            h.PremiseId premise_id,
            (cp.CPYearID - 1) year, cp.CPDate comed_cp_date,
            cp.ZonalLoad zonal_load, Avg(h.Usage) usage 
        from HourlyUsage h
        inner join COMED_CoincidentPeak cp
            on cp.CPDate = h.UsageDate AND
            cp.HourEnding = h.HourEnding
        where h.UtilityId = 'COMED'
        group by
            h.PremiseId, (cp.CPYearID - 1), cp.CPDate, cp.ZonalLoad
        order by h.PremiseId, cp.CPDate
        """
        return pd.read_sql(query, self.conn)

    def get_delivery_service_class(self):
        query = """
        select
            PremiseId premise_id,
            DSC dsc
        from Premise
        where UtilityId = 'COMED'
        """
        return pd.read_sql(query, self.conn)

    def get_normalized_peak_loads(self) -> pd.DataFrame:
        query = '''
            select 
                Cast((CPYearId - 1) as int) as Year,
                ParameterValue as NormalizedPeakLoad
            from [SystemLoad]
            where
                ParameterId = 'WeatherNormPeakLoad'
        '''
        return pd.read_sql(query, self.conn).set_index('Year')

    def get_cust_delta(self) -> pd.DataFrame:
        query = '''
                    select 
                        Cast((CPYearId - 1) as int) as Year,
                        ParameterValue as CustDelta
                    from [SystemLoad]
                    where
                        ParameterId = 'DeltaAvgComEdPJM'
                '''
        return pd.read_sql(query, self.conn).set_index('Year')

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
                --Avg(PJMZonalLoad) * 1000 as AvgCPZonalLoad
                Avg(COMEDZonalLoad) * 1000 as AvgCPZonalLoad
            from [CoincidentPeak]
            where UtilityId = 'COMED'
            group by
                Cast(CpYearId-1 as int)
            """
        df = pd.read_sql(query, conn).set_index('Year')
        df = pd.merge(df, self.normalized_peak_loads, left_index=True, right_index=True)
        # df['Step4Diff'] = self.normalized_peak_loads - df['AvgCPZonalLoad']
        df['Step4Diff'] = df['NormalizedPeakLoad'] - df['AvgCPZonalLoad']

        return df

    @staticmethod
    def get_pjm_cp_usage(conn: pymssql.Connection, premise: str = None) -> pd.DataFrame:
        """Select all records from COMED Hourly
        for PJM coincident peak usage. Filter those values that
        do not possess 5 values per year
        """

        # Updated query from Barsha 7/17/2017
        query = """
        select distinct
               h.PremiseId,
                iif(RTrim(p.RateClass)  is null, 'NULL', RTrim(p.RateClass)) as RateClass,
                RTrim(p.DSC) as DSC,
                CAST(cp.CPYearId -1 as INT) as Year,
                cp.CPDate as CPDatePJM,
                cp.HourEnding as CPHourEnding,
                --cp.HourEnding-1 as ADJCPHourEndingPJM,
                SUM(h.Usage) as UsagePJM  -- Changed by Barsha
            from [Premise] p
            inner join [HourlyUsage] h on
                p.PremiseId = h.PremiseId
            inner join [CoincidentPeak] cp on
                cp.UtilityId = h.UtilityId and
                cp.CPDate = h.UsageDate and
                cp.HourEnding = h.HourEnding
            where
                h.UtilityId = 'COMED'
                {prem}
            group by  -- Changed by Barsha
                h.PremiseId,
                iif(RTrim(p.RateClass)  is null, 'NULL', RTrim(p.RateClass)),
                RTrim(p.DSC),
                CAST(cp.CPYearId -1 as INT),
                cp.CPDate,
                cp.HourEnding
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
        #missing_data_index = grp[grp['CountPJM'] != 5.0].index
        missing_data_index = grp[grp['CountPJM'] == 0].index
        grp = grp.set_value(missing_data_index, 'MeanPJM', np.nan)

        return pd.merge(df, grp, how='left',
                        on=['PremiseId', 'Year'])

    
    @staticmethod
    def get_comed_cp_usage(conn: pymssql.Connection, premise: str = None) -> pd.DataFrame:
        """Select all records from COMED Hourly
        for COMED coincident peak usage. Filter those values that
        do not possess 5 values per year
        """

        query = """
        select distinct
                h.PremiseId,
                iif(RTrim(p.RateClass)  is null, 'NULL', RTrim(p.RateClass)) as RateClass,
                RTrim(p.DSC) as DSC,
                CAST(cp.CPYearId -1 as INT) as Year,
                cp.CPDate as CPDateCOMED,
                cp.HourEnding as CPHourEndingCOMED,
                ZonalLoad,
                SUM(h.Usage) as UsageCOMED -- Changed by Barsha
               
            from [HourlyUsage] h
            inner join [COMED_CoincidentPeak] cp on
                cp.UtilityId = h.UtilityId and
                cp.CPDate = h.UsageDate and
                cp.HourEnding = h.HourEnding
            inner join [Premise] p on
                p.PremiseId = h.PremiseId
            where
                h.UtilityId = 'COMED'
                {prem}
                     --/*
            group by   -- Changed by Barsha
                h.PremiseId,
                iif(RTrim(p.RateClass)  is null, 'NULL', RTrim(p.RateClass)),
                RTrim(p.DSC),
                CAST(cp.CPYearId -1 as INT),
                cp.CPDate,
                cp.HourEnding,
                ZonalLoad --*/
              
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
        #df.replace(to_replace=None, value="NULL", inplace=True)

        grp = df.groupby(['PremiseId', 'Year', 'RateClass'])['UsageCOMED'].agg(
            {'CountCOMED': len, 'MeanCOMED': np.mean})
        grp.reset_index(inplace=True)

        # set `Mean` = np.NaN if `Count` != 5
        missing_data_index = grp[grp['CountCOMED'] == 0].index
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
        return ((r.AcustPL - r.AcustCPL) / r.CustDelta)

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

        df = pd.merge(df, self.cust_delta, left_index=True, right_index=True)



        df['Step6'] = df.apply(self.step6, axis=1)
        df['Step7'] = df['Step6'] * df['Step4Diff']
    
        df['ICap'] = df.apply(self.icap, axis=1)
        df['Strata'] = df['DSC_x']

        df.rename(columns={'DSC_x':'RateClass'}, inplace=True)
        df.reset_index(inplace=True)
     
        self.compute_nits(df)


        df.to_csv(path_or_buf='/home/ubuntu/comed.csv', index=False)
        return meta_organize(self, df)

    def compute_nits(self, df):
        # Cleanup
        cmd_icap = df.copy()
        icap = cmd_icap.drop(labels=['DistLossFactor_y', 'TransLossFactor_y'], axis=1).copy()
        icap = icap.rename(columns={'DistLossFactor_x': 'DistLossFactor', 'TransLossFactor_x': 'TransLossFactor'})

    

        # Initialize records using PJM CP data
        records = list()
        for k, g in self.pjm_records.groupby(['premise_id', 'year']):
            r = Record(*k) # premise, year 
            r.pjm_cp_df = g.drop(labels=['premise_id', 'year'], axis=1).copy() 
            records.append(r)

        # Populate records with COMED CP data
        cmd_data = self.comed_records.copy()
        for r in records:
            prem_data = cmd_data[(cmd_data.premise_id == r.premise_id) & (cmd_data.year == r.year)]
            data = prem_data.drop(labels=['premise_id', 'year'], axis=1)
            r.comed_cp_df = data.copy()

        # Add Delivery Service Class
        for r in records:
            _dsc = self.dsc_records[self.dsc_records.premise_id == r.premise_id]['dsc'].values[0]
            r.dsc = _dsc

        # Add existing ICap Caluclations and factors
        for r in records:
            prem = r.premise_id
            year = r.year
            
            df = icap[(icap.PremiseId == prem) & (icap.Year == year)]
            df = df.drop(labels=['PremiseId', 'Year', 'RateClass', 'Strata'], axis=1).copy()
            r.icap_df = df       

        # Build string
        for r in records:
            r.string_builder()


        rw = RecordWriter(records)
        rw.write()
  


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
    #return df

class RecordWriter:
    def __init__(self, records=None):
        assert(records is not None)
        self.records = records
        
        self.filename = 'comed_interval_nits.csv'
        self.path = os.path.join('/home/ubuntu/JustEnergy/', self.filename)
        
        
    def write(self):
        self.write_header()
        self.write_records()
        
    def write_header(self):
        
        header = 'PREMISEID, DSC, RUNDATE,'\
        'PJM_CP_1,PJM_HOURENDING_1,PJM_USAGE_1,'\
        'PJM_CP_2,PJM_HOURENDING_2,PJM_USAGE_2,'\
        'PJM_CP_3,PJM_HOURENDING_3,PJM_USAGE_3,'\
        'PJM_CP_4,PJM_HOURENDING_4,PJM_USAGE_4,'\
        'PJM_CP_5,PJM_HOURENDING_5,PJM_USAGE_5,'\
        'COMED_CP_1,COMED_HOURENDING_1,COMED_USAGE_1,COMED_ZONAL_1,'\
        'COMED_CP_2,COMED_HOURENDING_2,COMED_USAGE_2,COMED_ZONAL_2,'\
        'COMED_CP_3,COMED_HOURENDING_3,COMED_USAGE_3,COMED_ZONAL_3,'\
        'COMED_CP_4,COMED_HOURENDING_4,COMED_USAGE_4,COMED_ZONAL_4,'\
        'COMED_CP_5,COMED_HOURENDING_5,COMED_USAGE_5,COMED_ZONAL_5,'\
        'DISTRIBUTION_LOSS,TRANSMISSION_LOSS,CUSTOMER_DELTA,UFT,UFC,ACUSTCPL,'\
        'ACUSTPL,ICAP,NITS'
        
        with open(self.path, 'w') as fout:
                fout.write(header + os.linesep)
        return
    
    def write_records(self):
        with open(self.path, 'a+') as fout:
            for r in self.records:
                    fout.write(r.string_record + os.linesep)
        return

class Record:
    def __init__(self, premise_id=None, year=None):
        # PJM CP Data
        self.premise_id = premise_id
        self.year = year
        self.run_date = datetime.now()
        self.pjm_cp_df = None
        
        
        # Comed CP Data
        self.comed_cp_df = None
        
        self.dsc = None  # Delivery Service Class
        self.string_record = None
        self.icap_df = None
        self.nits = None
    
    def compute_nits(self):
        assert(self.icap_df is not None)
        try:
            acustpl = self.icap_df.AcustPL.iloc[0]
            dist_loss = self.icap_df.DistLossFactor.iloc[0]
            trans_loss = self.icap_df.TransLossFactor.iloc[0]
            uft = self.icap_df.UFT.iloc[0]
            self.nits =  acustpl * dist_loss * trans_loss * uft
        except IndexError as e:
            self.nits = None
        
        # nspl = ?

    def append_empty_rows(self, df):
        # Get number of rows to add
        num_new_rows = 5 - df.shape[0]

        # Empty series to append dataframe
        empty = pd.Series([np.NaN for _ in range(df.shape[1])], index=df.columns, name='empty')
        for r in range(num_new_rows):
            df = df.append(empty)
        return df
                

    def format_df(self, df):
        if df.shape[0] > 5:
            return df.iloc[:5]
        elif df.shape[0] < 5:
            return self.append_empty_rows(df)
        else:
            return df
        
    def string_builder(self):
        assert(self.pjm_cp_df is not None)
        assert(self.comed_cp_df is not None)
        assert(self.icap_df is not None)
        if self.nits is None:
            self.compute_nits()
        
        if self.nits is None:
            self.string_record = "missing data"
            return
        rec = ''
        rec += '{premise_id}, {dsc}, {run_date},'.format(**self.__dict__)
        
        # PJM CP Data
        pjm = self.format_df(self.pjm_cp_df.sort_values(by='pjm_cp_date')) 
        for row in pjm.itertuples():
            _, cp, usage = row
            rec += '{cp}, {hour}, {usage},'.format(cp=cp, hour=None, usage=usage)
            
        # COMED CP Data
        comed = self.format_df(self.comed_cp_df.sort_values(by='comed_cp_date'))
        for row in comed.itertuples():
            _, cp, zonal, usage = row
            data = dict(cp=cp, hour=None, usage=usage, zonal=zonal)
            rec += '{cp}, {hour}, {usage}, {zonal},'.format(**data)
            
        # ICap Data
        rec += '{},'.format(self.icap_df.DistLossFactor.iloc[0])
        rec += '{},'.format(self.icap_df.TransLossFactor.iloc[0])
        rec += '{},'.format(self.icap_df.CustDelta.iloc[0])
        rec += '{},'.format(self.icap_df.UFT.iloc[0])
        rec += '{},'.format(self.icap_df.UFC.iloc[0])
        rec += '{},'.format(self.icap_df.AcustCPL.iloc[0])
        rec += '{},'.format(self.icap_df.AcustPL.iloc[0])
        rec += '{},'.format(self.icap_df.ICap.iloc[0])
        rec += '{}'.format(self.nits)
        
            
        self.string_record = rec
