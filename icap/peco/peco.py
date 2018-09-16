import pandas as pd
import numpy as np
from datetime import datetime


import os
import tempfile

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
        sys = self.sys_df_.copy()

        # BEGIN PREPROCESSING
        # 1. Obtain index values for required utility params
        # 2. Merge records with correct NCRatio factor

        # 1. INDEX
        # index values for RateClassLoss and NCRatio
        rc_idx = util[util['ParameterId'] == 'RateClassLoss'].index
        nc_idx = util[util['ParameterId'] == 'NCRatio'].index
        nspl = sys[sys['ParameterId'] == 'TransLoadScaleFactor']
        nspl = nspl.rename(columns={'ParameterValue': 'NSPLScale'}).drop(labels=['ParameterId'], axis=1)
        

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
        #rec.to_csv('/tmp/pecointerval.csv', index=False)
        grp = rec.groupby(['PremiseId', 'Year', 'RateClass', 'Strata']
                          )['WCF'].agg({'Count': len, 'WCF': np.mean}).reset_index()

        rec['WCFMean'] = rec.groupby(['PremiseId', 'Year', 'RateClass', 'Strata'])['WCF'].transform(np.mean)
        rec['Count'] = rec.groupby(['PremiseId', 'Year', 'RateClass', 'Strata'])['WCF'].transform(len)
        grp = rec.copy()

        self.factors_ = rec 
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
        _tmp = pd.merge(grp, rate_class,
                       on=['Year', 'RateClass', 'Strata'], how='left')
        tmp = pd.merge(_tmp, nspl, on='Year')
        # END RATECLASSLOSS

        # ICAP
        # icap = wcf * rate_class_loss_factor ; (ParameterValue)
        def icap_calc(group):
            return group['WCFMean'].mean() * group['ParameterValue'].mean()
        
        icap = tmp.groupby(['PremiseId', 'Year', 'RateClass', 'Strata']).apply(icap_calc).reset_index()
        icap = icap.rename(columns={0:'ICap'})

        wicap = pd.merge(tmp, icap, on=['PremiseId', 'Year', 'RateClass', 'Strata'])
        wicap['NITS'] = wicap['ICap'] * wicap['NSPLScale']
        wicap = wicap.rename(columns={'ParameterValue': 'RCLF'})
        
        tmp['ICap'] = tmp['WCF'] * tmp['ParameterValue']
        #tmp['NITS'] = tmp['ICap'] * tmp['NSPLScale']
        
        #self.icap_df_ = tmp
        self.icap_df_ = wicap


        # Write to nits;
        # Results object returned from self.compute_icap() causes
        # error in main.py
        self.write_nits()

        return meta_organize(self, tmp)

    def write_nits(self):

        nspl_scale = self.sys_df_[self.sys_df_['ParameterId'] == 'TransLoadScaleFactor']
        nspl_scale = nspl_scale.rename(columns={'ParameterValue': 'NSPLScale'}).drop(labels=['ParameterId'], axis=1)

        # Extract parameter values
        nc_ratio = self.util_df_[self.util_df_['ParameterId'] == 'NCRatio'].copy()
        rclf = self.util_df_[self.util_df_['ParameterId'] == 'RateClassLoss']
        util_min = pd.merge(nc_ratio, rclf, on=['Year', 'CPDate', 'RateClass', 'Strata'], how='left').fillna(
            method='ffill')

        # Join hourly records and parameter values
        _plc = (pd.merge(self.records_, util_min,
                         left_on=['UsageDate', 'RateClass', 'Strata', 'Year'],
                         right_on=['CPDate', 'RateClass', 'Strata', 'Year'],
                         how='left')
                .rename(columns={'ParameterValue_x': 'NCRatio', 'ParameterValue_y': 'RCLF'})
                .drop(labels=['ParameterId_x', 'ParameterId_y'], axis=1))
        plc = pd.merge(_plc, nspl_scale, on=['Year'])

        # Compute
        plc['PLC_factor'] = plc.Usage * plc.NCRatio * plc.RCLF
        plc['PLC'] = plc.groupby(['PremiseId', 'Year'])['PLC_factor'].transform(np.mean)
        plc['NITS'] = plc.PLC * plc.NSPLScale

        write_nits_to_csv(plc, 'INT')
        return


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
        rec['DmdAvg'] = rec.groupby(['PremiseId', 'Year'])['Demand'].transform('mean')
        rec['DMDCount'] = rec.groupby(['PremiseId', 'Year'])['Demand'].transform('count')

	
        redrec = rec.copy()
        #redrec = rec.groupby(['PremiseId', 'Year']
        #                     )['Demand'].agg({'Count': len, 'DmdAvg': np.mean}).reset_index()

        # set insufficent records to NaN
        # if count != 4 or dmdavg == 0 then dmdavg -> np.nan

        #bad_rec_idx = redrec[(redrec['Count'] > 0) |
        #                     (redrec['DmdAvg'] == 0.0)].index
        #redrec.set_value(bad_rec_idx, 'DmdAvg', np.nan)

        # 3.
        #nared = util.ix[na_idx].groupby(['Year', 'RateClass', 'Strata']
        #                                )['ParameterValue'].agg(
        #    {'Count': len, 'NAAvg': np.mean}
        #).reset_index()
	
        nared = util[util['ParameterId'] == 'NARatio'].copy()
        nared['NAAvg'] = nared.groupby(['Year', 'RateClass', 'Strata'])['ParameterValue'].transform('mean')
        nared['NACount'] = nared.groupby(['Year', 'RateClass', 'Strata'])['ParameterValue'].transform('count')
        nared.reset_index(inplace=True)
        # set insufficent records to NaN
        # if count != 5 or naavg == 0 then naavg -> np.nan
        #bad_na_idx = nared[(nared['Count'] > 0) |
        #                   (nared['NAAvg'] == 0.0)].index
        #nared.set_value(bad_na_idx, 'NAAvg', np.nan)

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

        #
        self.compute_nits()

        #return nared
        return meta_organize(self, tmp_3.drop_duplicates())

    def compute_nits(self):
        # Values for the PLC/ICAP
        records = self.records_.copy()
        util = self.util_df_.copy()
        sys = self.sys_df_.copy()

        # Transmission scaling factor for NITS
        nspl_scale = sys[sys['ParameterId'] == 'TransLoadScaleFactor'].copy()
        nspl_scale = nspl_scale.rename(columns={'ParameterValue': 'NSPLScale'}).drop(labels='ParameterId', axis=1)

        # NA Ratio
        # Average the NA ratio by [Year, RateClass, Strata]
        na_ratio = util[util['ParameterId'] == 'NARatio'].copy()
        na_ratio = na_ratio.rename(columns={'ParameterValue': 'NARatio'}).drop(labels='ParameterId', axis=1)
        na_ratio['NAAvg'] = na_ratio.groupby(['Year', 'RateClass', 'Strata']).transform('mean')

        # Rate Class Loss Factor
        rclf = util[util['ParameterId'] == 'RateClassLoss'].copy()
        rclf = rclf.rename(columns={'ParameterValue': 'RCLF'}).drop(labels='ParameterId', axis=1)

        # PLCScale
        plcf = sys[sys['ParameterId'] == 'PLCScaleFactor'].copy()
        plcf = plcf.rename(columns={'ParameterValue': 'PLCScaleFactor'}).drop(labels='ParameterId', axis=1)

        # Merge the utility values; fill missing with NaN
        # Forward fill fills by year and rateclass
        util_min = pd.merge(na_ratio, rclf, on=['Year', 'CPDate', 'RateClass', 'Strata'], how='left'
                            ).fillna(method='ffill')

        util_min = pd.merge(util_min, nspl_scale, on='Year', how='left')
        util_min = pd.merge(util_min, plcf, on='Year', how='left')


        # Average demand per premise per year
        records['AvgDmd'] = records.groupby(['PremiseId', 'Year', 'RateClass', 'Strata'])['Demand'].transform('mean')
        records['RecCount'] = records.groupby(['PremiseId', 'Year', 'RateClass', 'Strata'])['Demand'].transform('count')

        # Group records
        objs = list()
        for k, v in records.groupby(['PremiseId', 'Year', 'RateClass', 'Strata']):
            r = Record(*k)

            # Pad to ensure demand is always 4
            demand = v['Demand'].values
            demand = np.pad(demand, (0, 4 - len(demand)), 'constant', constant_values=np.NaN)
            r.demand = ','.join(str(x) for x in demand)
            r.avg_demand = v['AvgDmd'].values[0]
            objs.append(r)

        for r in objs:
            # Filter params by record object
            c1 = util_min['Year'] == r.year
            c2 = util_min['Strata'] == r.strata
            c3 = util_min['RateClass'] == r.rateclass
            _df = util_min[c1 & c2 & c3]



            # Pad to enusre ratios always have 5 values
            na_ratios = _df['NARatio'].values
            na_ratios = np.pad(na_ratios, (0, 5 - len(na_ratios)), 'constant', constant_values=np.NaN)
            r.na_ratio = ','.join(str(x) for x in na_ratios)
            try:
                r.na_avg = _df['NAAvg'].values[0]
            except IndexError as e:
                r.na_avg = np.nan
            
            try:
                r.rclf = _df['RCLF'].values[0]
            except IndexError as e:
                r.rclf = np.nan
            
            try:
                r.nspl_scale = _df['NSPLScale'].values[0]
            except IndexError as e:
                r.nspl_scale = np.nan

            try:
                r.plcf = _df['PLCScaleFactor'].values[0]
            except IndexError as e:
                r.plcf = np.nan

            r.compute()

        rw = RecordWriter(objs)
        rw.write()

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


def write_nits_to_csv(df, meter_type):

    run_date = datetime.now()
    fileout = 'peco_interval_nits.csv'
    with open(fileout, 'w') as fout:
        # Header
        fout.write('PremiseId, RateClass, Strata, Year, RunDate,\
        Usage Date 1 (CP 1), Hour Ending 1 (CP 1), CP 1 Usage, NCRatio,\
        Usage Date 2 (CP 2), Hour Ending 2 (CP 2), CP 2 Usage, NCRatio,\
        Usage Date 3 (CP 3), Hour Ending 3 (CP 3), CP 3 Usage, NCRatio,\
        Usage Date 4 (CP 4), Hour Ending 4 (CP 4), CP 4 Usage, NCRatio,\
        Usage Date 5 (CP 5), Hour Ending 5 (CP 5), CP 5 Usage, NCRatio,\
        Rate Class Loss, Meter Type, Capacity Planning Year, PLC/ICAP, NSPL Scale, NITS\n')


        # Loop the groups
        df['Year'] = df['Year'].apply(lambda x: str(int(x) + 1))
        for name, group in df.groupby(['PremiseId']):

            # If group is missing rows, force empty rows for valid print lengths
            if group.shape[0] != 5:
                num_empties = 5 - group.shape[0]
                empty_row = pd.Series([np.nan for col in group.columns], index=group.columns)

                for _ in range(num_empties):
                    group = group.append(empty_row, ignore_index=True)

            # Updated group shape (should be 5)
            last_row = group.shape[0] - 1
            row_number = 0
            current_row = ''
            # Build string output for each group
            for row in group.sort_values(by=['CPDate']).itertuples():
                # Unpack the row
                index, premise_id, usage_date, hour_ending, usage, year, rate_class, strata, _, nc_ratio, rclf, nspl_scale, _, plc, nits = row
                # index, premise_id, rate_class, strata, year, usage_date, hour_ending, usage, nc_ratio, wcf, wcf_mean, rec_count, rclf, icap, run_date, meter_type = row

                # Write initial fields
                if row_number == 0:
                    current_row = '{},{},{},{},{},'.format(
                        premise_id, rate_class, strata, year, run_date)

                # Build repeated fields
                current_row += '{},{},{},{},'.format(usage_date, hour_ending, usage, nc_ratio)

                # Final fields
                if row_number == last_row:
                    current_row += '{},{},{},{},{},{}\n'.format(rclf, meter_type, year, plc, nspl_scale, nits)

                row_number += 1

            # Write current group
            fout.write(current_row)


class RecordWriter:
    def __init__(self, records=None):
        assert (records is not None)
        self.records = records

    def write(self, fp=None):
        if os.path.exists('/home/ubuntu/JustEnergy'):
            fp = 'peco_demand_nits.csv'

        elif fp is None:
            fp = os.path.join(tempfile.gettempdir(), 'peco_demand_nits.csv')

        else:
            fp = 'peco_demand_nits.csv'


        header = 'PREMISE,RATECLASS,STRATA,YEAR,RUNDATE,NA1,NA2,NA3,NA4,NA5,NAAVG,'\
                 'DMD1,DMD2,DMD3,DMD4,DMDAVG,RCLF,PLCF,PLC,NSPLSCALE,NITS'

        with open(fp, 'w') as fout:
            fout.write(header + os.linesep)
            for record in self.records:
                fout.write(record.string_record + os.linesep)


# Record object to aggregate the information iteratively
class Record:
    def __init__(self, premise_id, year, rateclass, strata):
        self.premise_id = premise_id
        self.year = year
        self.plcyear = str(int(year) + 1)
        self.rateclass = rateclass
        self.strata = strata

        self.demand = None
        self.avg_demand = None

        self.na_ratio = None
        self.na_avg = None
        self.rclf = None
        self.nspl_scale = None
        self.plcf = None
        self.rundate = datetime.now()

        self.plc = None
        self.nits = None
        self.computed = False
        self.string_record = None

    def __repr__(self):
        return 'Record<id={premise_id}, year={year}, rc={rateclass}, strata={strata}>'.format(**self.__dict__)

    def string_builder(self):
        assert (self.computed)
        return '{premise_id},{rateclass},{strata},{plcyear},{rundate},{na_ratio},{na_avg},' \
               '{demand},{avg_demand},{rclf},{plcf},{plc},{nspl_scale},{nits}'.format(**self.__dict__)

    def compute(self):
        assert (self.avg_demand is not None)
        assert (self.na_avg is not None)
        assert (self.rclf is not None)
        assert (self.nspl_scale is not None)
        assert (self.plcf is not None)
        self.plc = self.avg_demand * self.na_avg * self.rclf * self.plcf
        self.nits = self.plc * self.nspl_scale

        self.computed = True
        self.string_record = self.string_builder()
