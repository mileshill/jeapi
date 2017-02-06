import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from scipy.stats.stats import pearsonr
from scipy.stats import linregress

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)


class CorrelationException(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class Premise():
    '''Load a premise and normalize data over given year'''

    def __init__(self, conn=None, year=None, utility=None, premise=None):
        if not all((premise, conn, utility, year)):
            raise CorrelationException(
                (conn, year, utility, premise), 'None value(s)')

        # instantiation vars
        self.conn = conn
        self.year = year
        self.utility = utility.upper()
        self.id = premise

        # computational vars
        self.history = None

    def __repr__(self):
        '''Defines the printable output'''
        params = (self.year, self.utility, self.id, not self.history_.empty)
        rep = "Premise(year=%d, utility=%s, id=%s, initialized=%s)"

        return rep % params

    def initialize(self):
        '''Loads entire year of usage data for every day/hour possible.
        RETURN: pd.DataFrame
        columns = ['UsageDate', 'HourEnding', 'Usage', 'NormalizedUsage']
        '''
        usage_query = """
            select
                Cast(UsageDate as datetime) as UsageDate,
                HourEnding,
                Usage as PremUsage
            from HourlyUsage
            where
                Year(UsageDate) = {year} and
                UtilityId = '{utility}' and
                PremiseId = '{id}'
            order by UsageDate, HourEnding
        """

        # load query results into pd.Dataframe
        params = {'year': self.year, 'utility': self.utility, 'id': self.id}
        df = pd.read_sql(usage_query.format(**params), self.conn)

        # normalize the usage to range [0.0, 1.0]
        df['PremNormalizedUsage'] = MinMaxScaler().fit_transform(
            df.PremUsage.values.reshape(-1, 1))

        df = flag_peak_days(df)
        df = flag_cp_days(self.conn, self.utility, df)

        # modify timestamp
        #   UsageDate = YYYY-MM-DD 00:00:00
        #   HourEnding = integer
        #   new UsageDate = DoW(abr) Month(abr) Day(padded) YYYY HR:MM:SS
        df['HourEnding'] = df['HourEnding'].apply(
            lambda x: pd.Timedelta(hours=x))
        df['UsageDate'] = (df['UsageDate'] + df['HourEnding']).apply(
            lambda x: x.strftime('%a %b %d %Y %H:%M:%S'))
        df.drop('HourEnding', axis=1, inplace=True)
        # object modifies state
        self.history_ = df


class ZonalLoad():
    '''Select usage information at the utility/iso level.

    CURRENTLY SELECTS ONLY FROM PJMHOURLYLOADS TABLE
    '''

    def __init__(self, conn, year=None, zone=None):
        if not all((year, zone)):
            raise CorrelationException((year, zone), 'None value(s)')

        # instantiation vars
        self.conn = conn
        self.year = year
        self.zone = zone

        # computational vars
        self.history_ = None

    def initialize(self):
        usage_query = """
            select
                UsageDate,
                Cast(Right(HourEnding, 2) as float) as HourEnding,
                Usage as ZoneUsage
            from PJMHourlyLoads
            where
                LoadArea = '{zone}' and
                Year(UsageDate) = {year}
            order by UsageDate, HourEnding
        """
        # load data for year and load area
        params = {'year': self.year, 'zone': self.zone}
        df = pd.read_sql(usage_query.format(**params), self.conn)

        # normalize usage to given range [0.0, 1.0]
        df['ZoneNormalizedUsage'] = MinMaxScaler().fit_transform(
            df.ZoneUsage.values.reshape(-1, 1))

        df = flag_peak_days(df)

        df = flag_cp_days(self.conn, self.zone, df)
        # modify timestamp
        #   UsageDate = YYYY-MM-DD 00:00:00
        #   HourEnding = integer
        #   new UsageDate = YYYY-MM-DD HH:00:00
        df['HourEnding'] = df['HourEnding'].apply(
            lambda x: pd.Timedelta(hours=x))
        df['UsageDate'] = (df['UsageDate'] + df['HourEnding']).apply(
            lambda x: x.strftime('%a %b %d %Y %H:%M:%S'))
        df.drop('HourEnding', axis=1, inplace=True)
        # object modifies state
        self.history_ = df

        # update the history to shop `top` peak days
        self.history_ = df


class CorrelationResult():
    '''CorrelationResult:
    Holds the data to be returned to user. Created for simplyfing the returned
    object and tracking results.
    '''

    def __init__(self, year=None, iso=None, utility=None, premise=None):
        if not all((year, iso, utility, premise)):
            raise CorrelationException(
                (year, iso, utility, premise), 'None value(s)')

        self.year = year
        self.iso = iso
        self.utility = utility
        self.premise = premise
        self.failed = False

    def results(self):
        return self.__dict__


class Correlation():
    '''CORRELATION ANALYSIS
    Purpose:
        The correlation analysis compares computes the PEARSON-R and R^2 value
        for a given PREMISE-[UTILITY, ISO] combination. These metrics assess
        how closely a PREMISE's behaviour corresponds to the various providers.
    '''

    def __init__(self, conn, year=None, iso=None, utility=None, premise=None):

        if not all((year, iso, utility, premise)):
            raise CorrelationException(
                (year, iso, utility, premise), 'None value(s)')

        # instantiation vars
        self.conn = conn
        self.year = year
        self.iso = iso
        self.utility = utility
        self.premise = premise

    def analyze(self):
        '''self.analyze
        Purpose:
            Executes the correlation analysis.

        Method:
            1. Initialization
                a. instantiate objects for iso, utility, and premise.
                b. initialize objects; executes SQL query and
                    returns pd.DataFrame
            2. Preprocessing
                a. merge data frames
            3. Absolute Difference
                a. abs(utility_normalized_usage - premise_normalized_usage)
            4. Statistical Description
                a. pearsonr(x_1 ,x_2)
                b. rsqr(x_1, x_2)
            5. Result log
                a. store results in CorrelationResult object

        RETURN:
            cls CorrelationResult

        '''
        conn = self.conn
        year, iso, util, premise = self.year, self.iso, self.utility, self.premise
        results = CorrelationResult(
            year=year, iso=iso, utility=util, premise=premise)

        # BEGIN INITIALIZATION
        # initialize three levels
        iso = ZonalLoad(conn, year=year, zone=iso)  # iso.initialize()
        utility = ZonalLoad(conn, year=year, zone=util)  # utility.initialize()
        prem = Premise(conn, year=year, utility=util,
                       premise=premise)  # prem.initialize()
        zones = [iso, utility, prem]
        for zone in zones:
            zone.initialize()       # executes SQL query

        # if no records, halt evaluation
        if prem.history_.empty:
            results.failed = True
            return resultsi
        # END INITIALIZATION

        # BEGIN PREPROCESSING
        # preprocessing: join data frames on=['UsageDate', 'HourEnding']
        join_cols = ['UsageDate']
        prem_util = pd.merge(prem.history_, utility.history_, on=join_cols)
        prem_iso = pd.merge(prem.history_, iso.history_, on=join_cols)
        util_iso = pd.merge(utility.history_, iso.history_, on=join_cols)
        # END PREPROCESSING

        # BEGIN ABSOLUTE DIFFERENCE
        # absolute difference between premise and utility
        # this value is used to understand the behavioural variations
        prem_util['Delta'] = np.abs(
            prem_util.ZoneNormalizedUsage - prem_util.PremNormalizedUsage)

        # make a copy
        temp = prem_util.copy()

        # keep necessary columns
        cols = ['PremNormalizedUsage', 'PremUsage',
                'ZoneNormalizedUsage', 'ZoneUsage',
                'PeakDay_x', 'PeakDay_y']

        keep = ['UsageDate', 'Delta']
        temp.drop(cols,
                  axis=1, inplace=False)

        # assign to CorrelationResults object
        results.delta_record = temp[keep].to_dict('records')
        del temp
        # END ABSOLUTE DIFFERENCE

        # BEGIN STATISTICAL DESCRIPTION
        # obtain Pearsonr value
        results.pearson_prem_util = pearsonr(
            prem_util.PremNormalizedUsage, prem_util.ZoneNormalizedUsage)[0]
        results.pearson_prem_iso = pearsonr(
            prem_iso.PremNormalizedUsage, prem_iso.ZoneNormalizedUsage)[0]
        results.pearson_util_iso = pearsonr(
            util_iso.ZoneNormalizedUsage_x, util_iso.ZoneNormalizedUsage_y)[0]

        # r^2 value
        # slop, intercept, r-value, p-value, stderr = linregress(x,y)
        # rsqr = r-value ** 2
        results.rsqr_prem_util = linregress(
            prem_util.PremNormalizedUsage, prem_util.ZoneNormalizedUsage)[2]**2
        results.rsqr_prem_iso = linregress(
            prem_iso.PremNormalizedUsage, prem_util.ZoneNormalizedUsage)[2]**2
        results.rsqr_util_iso = linregress(
            util_iso.ZoneNormalizedUsage_x, util_iso.ZoneNormalizedUsage_y)[2]**2
        # END STATISTICAL DESCRIPTION

        # BEGIN RESULT LOG
        # log usage normalized history for plotting
        results.premise_record = prem.history_.to_dict('records')
        results.utility_record = utility.history_.to_dict('records')
        results.iso_record = iso.history_.to_dict('records')
        # END RESULT LOG

        # RETURN CorrelationResult object
        return results


def flag_peak_days(df, top=20):
    '''Sets `PeakDay` = True if `NormalizedUsage` is in top 20 values.
    Values are considered on a per day basis.
    '''

    # call the history
    # df = pd.DataFrame(self.history_)

    # get the normalized usage column name;
    #    ['PremNormalizedUsage, ZoneNormalizedUage]
    col = [c for c in df.columns if 'NormalizedUsage' in c][0]

    # index values of top (n) NormalizedUsage values
    idx = df.groupby('UsageDate')[col].transform(max) == df[col]

    # select top values
    peak_values = df[idx].sort_values(by=col, ascending=False)[:top].index

    # add `PeakDay` column to df and assign values
    PEAK_DAY = 'PeakDay'
    df[PEAK_DAY] = False
    for index in peak_values:
        df.set_value(index, PEAK_DAY, True)

    # update the history to shop `top` peak days
    # self.history_ = df
    return df

def flag_cp_days(conn, utility, df):
    cp_query = """
        select
            CPDate,
            cast(HourEnding as int) as HourEnding
        from [CoincidentPeak]
        where
            UtilityId = '{utility}'"""

    cp_query = cp_query.format(utility=utility)
    cp_df = pd.read_sql(cp_query, conn)
    cp_df['CPDate'] = pd.to_datetime(cp_df['CPDate'])

    cp_idx = df.reset_index().merge(cp_df,
        how='inner',
        left_on=['UsageDate', 'HourEnding'],
        right_on=['CPDate', 'HourEnding']).set_index('index').index

    df['CoincidentPeak'] = 0
    df.set_value(cp_idx, 'CoincidentPeak', 1)

    return df
