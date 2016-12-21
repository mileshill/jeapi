import pandas as pd
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
            raise CorrelationException((conn, year, utility, premise), 'None value(s)')

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
        params = {'year':self.year, 'utility':self.utility, 'id':self.id}
        df = pd.read_sql(usage_query.format(**params), self.conn)

        # normalize the usage to range [0.0, 1.0]
        df['PremNormalizedUsage'] = MinMaxScaler().fit_transform(
               df.PremUsage.values.reshape(-1,1))

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
        params = {'year':self.year, 'zone':self.zone.upper()}
        df = pd.read_sql(usage_query.format(**params), self.conn)
        # normalize usage to given range [0.0, 1.0]
        df['ZoneNormalizedUsage'] = MinMaxScaler().fit_transform(
                df.ZoneUsage.values.reshape(-1,1))
        self.history_ = df

class CorrelationResult():
    def __init__(self, year=None, iso=None, utility=None, premise=None):
        if not all((year, iso, utility, premise)):
            raise CorrelationException((year, iso, utility, premise), 'None value(s)')


        self.year = year
        self.iso = iso
        self.utility = utility
        self.premise = premise
        self.failed = False
    def results(self):
        return self.__dict__

class Correlation():
    '''Executes correlation analysis

    RETURNS CorrelationResult object
    '''
    def __init__(self, conn, year=None, iso=None, utility=None, premise=None):

        if not all((year, iso, utility, premise)):
           raise CorrelationException((year, iso, utility, premise), 'None value(s)')

        # instantiation vars
        self.conn = conn
        self.year = year
        self.iso = iso
        self.utility = utility
        self.premise = premise

    def analyze(self):
        conn = self.conn
        year, iso, util, premise = self.year, self.iso, self.utility, self.premise
        results = CorrelationResult(year=year, iso=iso, utility=util, premise=premise)

        # initialize three levels
        iso = ZonalLoad(conn, year=year, zone=iso); iso.initialize()
        utility = ZonalLoad(conn, year=year, zone=util); utility.initialize()
        prem = Premise(conn, year=year, utility=util, premise=premise); prem.initialize()

        # if no records, halt evaluation
        if prem.history_.empty:
            results.failed = True
            return results

        # preprocessing: join data frames on=['UsageDate', 'HourEnding']
        join_cols = ['UsageDate', 'HourEnding']
        prem_util = pd.merge(prem.history_, utility.history_, on=join_cols)
        prem_iso = pd.merge(prem.history_, iso.history_, on=join_cols)
        util_iso = pd.merge(utility.history_, iso.history_, on=join_cols)

        # obtain Pearsonr value
        results.pearson_prem_util = pearsonr(prem_util.PremNormalizedUsage, prem_util.ZoneNormalizedUsage)[0]
        results.pearson_prem_iso  = pearsonr(prem_iso.PremNormalizedUsage, prem_iso.ZoneNormalizedUsage)[0]
        results.pearson_util_iso  = pearsonr(util_iso.ZoneNormalizedUsage_x, util_iso.ZoneNormalizedUsage_y)[0]

        # r^2 value
        # slop, intercept, r-value, p-value, stderr = linregress(x,y)
        # rsqr = r-value ** 2
        results.rsqr_prem_util = linregress(prem_util.PremNormalizedUsage, prem_util.ZoneNormalizedUsage)[2]**2
        results.rsqr_prem_iso  = linregress(prem_iso.PremNormalizedUsage, prem_util.ZoneNormalizedUsage)[2]**2
        results.rsqr_util_iso  = linregress(util_iso.ZoneNormalizedUsage_x, util_iso.ZoneNormalizedUsage_y)[2]**2

        # log usage normalized history for plotting
        results.premise_record = prem.history_.to_dict('records')
        results.utility_record = utility.history_.to_dict('records')
        results.iso_record = iso.history_.to_dict('records')

        # return CorrelationResult object
        return results








