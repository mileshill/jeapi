import pandas as pd
import numpy as np
from datetime import datetime
import os
import tempfile

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
                h.HourEnding,
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

        self.tmp = tmp
        icap = tmp.groupby(['PremiseId', 'Year', 'RateClass']
                           ).apply(ppl_icap).reset_index()

        icap.rename(columns={0: 'ICap'}, inplace=True)

        icap['Strata'] = 'NULL'

        self.icap_df = icap

        # Run the nits calc
        self.compute_nits()
        return meta_organize(self, icap)

    def compute_nits(self):
        # Records, Utility and System
        records = self.records_.copy()
        records = records.rename(columns={'UsageDate': 'CPDate'})
        util = self.util_df_.copy()
        sys = self.sys_df_.copy()

        # Extract reconcillation factors
        sys['CPDate'] = sys['ParameterId'].apply(lambda s: s.split()[0])
        sys = sys.rename(columns={'ParameterValue': 'ReconFactor'}).drop(labels='ParameterId', axis=1)

        # Loss factor
        loss = util[util['ParameterId'] == 'Loss Factor'].copy()
        loss = loss.rename(columns={'ParameterValue': 'LossFactor'}).drop(labels='ParameterId', axis=1)

        # Initialize all records
        objs = list()
        for k, g in records.groupby(['PremiseId', 'Year', 'RateClass']):
            r = Record(*k)
            r.cp_df = g[['Year', 'CPDate', 'HourEnding', 'Usage', 'RateClass']].sort_values(by='CPDate')
            objs.append(r)

        # Join on system params for each object
        for obj in objs:
            obj.cp_df = pd.merge(obj.cp_df, sys, on=['Year', 'CPDate'], how='left')
            obj.cp_df = pd.merge(obj.cp_df, loss, on=['Year', 'RateClass', 'CPDate'], how='left')
            obj.cp_df['LossFactor'] = obj.cp_df['LossFactor'].mean()

            obj.compute_plc()
            obj.string_builder()

        rw = RecordWriter(objs)
        rw.write()


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


class RecordWriter:
    def __init__(self, records=None):
        assert (records is not None)
        self.records = records

    def write(self, fp=None):
        if os.path.exists('/home/ubuntu/JustEnergy'):
            fp = '/home/ubuntu/JustEnergy/ppl_interval_nits.csv'
        elif fp is None:
            fp = os.path.join(tempfile.gettempdir(), 'ppl_interval_nits.csv')
        else:
            fp = os.path.join(os.path.abspath(__file__), 'ppl_interval_nits.csv')

        header = 'PREMISEID,RATECLASS,RUNDATE,' \
                 'CP_1,HOUR_ENDING_1,CP_1_USAGE,' \
                 'CP_2,HOUR_ENDING_2,CP_2_USAGE,' \
                 'CP_3,HOUR_ENDING_3,CP_3_USAGE,' \
                 'CP_4,HOUR_ENDING_4,CP_4_USAGE,' \
                 'CP_5,HOUR_ENDING_5,CP_5_USAGE,' \
                 'LOSS_FACTOR_1,LOSS_FACTOR_2,LOSS_FACTOR_3,LOSS_FACTOR_4,LOSS_FACTOR_5,' \
                 'RECON_FACTOR_1,RECON_FACTOR_2,RECON_FACTOR_3,RECON_FACTOR_4,RECON_FACTOR_5,' \
                 'METER_TYPE,' \
                 'CAPACITY_PLANNNG_YEAR,' \
                 'PLC,' \
                 'NITS'

        with open(fp, 'w') as fout:
            fout.write(header + os.linesep)
            for rec in self.records:
                fout.write(rec.string_record + os.linesep)

class Record:
    def __init__(self, premise_id=None, year=None, rateclass=None, strata=None):
        assert (premise_id is not None)
        assert (year is not None)
        assert (rateclass is not None)

        self.premise_id = premise_id
        self.year = year
        self.plcyear = str(int(year) + 1)
        self.rateclass = rateclass
        self.strata = strata
        self.rundate = datetime.now()

        self.cp_df = None
        self.plc = None
        self.nits = None

        self.string_record = None

    def compute_plc(self):
        assert (self.cp_df is not None)

        # Add empty rows where missing
        # Set plc to NaN; required 5 values
        if self.cp_df.shape[0] < 5:
            self.plc = np.NaN

            # Get number of rows to add
            num_new_rows = 5 - self.cp_df.shape[0]

            # Empty series to append dataframe
            empty = pd.Series([np.NaN for _ in range(self.cp_df.shape[1])], index=self.cp_df.columns, name='empty')
            for r in range(num_new_rows):
                self.cp_df = self.cp_df.append(empty)
            return

        # Compute PLC
        self.plc = (self.cp_df['Usage'] * self.cp_df['ReconFactor'] * self.cp_df['LossFactor']).mean()

    def compute_nits(self):
        raise NotImplementedError

    def __repr__(self):
        return 'Record<premise={premise_id}, rateclass={rateclass}, strata={strata}, year={year}>'.format(
            **self.__dict__)

    def string_builder(self):

        # Id, rateclass, rundate
        rec = '{premise_id},{rateclass},{rundate},'.format(**self.__dict__)

        # coincident peak date, hourending, usage
        for row in self.cp_df[['CPDate', 'HourEnding', 'Usage']].itertuples():
            _, cp, hour, usage = row
            rec += '{},{},{},'.format(cp, hour, usage)

        # Loss
        rec += ','.join(str(x) for x in self.cp_df['LossFactor'].values.tolist())
        rec += ','

        # Recon
        rec += ','.join(str(x) for x in self.cp_df['ReconFactor'].values.tolist())

        # Meter
        rec += ',INT'

        # Year
        rec += ',{}'.format(self.plcyear)

        # PLC and NITS
        rec += ',{plc},{nits}'.format(**self.__dict__)
        self.string_record = rec