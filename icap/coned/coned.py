import pandas as pd
import numpy as np
from datetime import datetime


class CONED:
    '''SuperClass for meter types. Loads the system and
    utility parameters from DB.
    Computes the LoadShapeAdjustmentTable from TemperatureVariant table.
    The LoadShapeAdjustmentTable is filtered per premise to create the
    LoadProfile
    '''

    def __init__(self, conn):
        self.params = {'ISO': 'NYISO',
                       'RunDate': datetime.now(),
                       'Utility': 'CONED'}

        # db connection
        self.conn = conn

        # CONED coincident peaks
        self.cp = self.get_coincident_peak()

        # rate class mapping for LoadShapeAdjustment calculation
        self.rc_map = self.get_rate_class_map()

        # utility parameters
        self.util = self.get_utility_params()

        # time of day codes
        self.tod_map = self.get_tod_map()

        # construct temperature variant table
        self.temp_var = self.build_temperature_variant()

        # construct load shape table from temperature variant table
        self.lst = self.build_lst()

    def get_utility_params(self):
        utility_query = """
            select
                CPYear-1 as Year,
                ParameterId,
                Replace(Zone, ' ', '') as Zone,
                Factor,
                case
                    when MeterType = 'Interval' then 'INT'
                    when MeterType = 'Demand' then 'DMD'
                    when MeterType = 'Scalar' then 'CON'
                    when MeterType like 'All%' then 'ALL'
                    else MeterType
                end as MeterType
            from
                CONED_UtilityParameters"""

        return pd.read_sql(utility_query, self.conn)

    def get_rate_class_map(self):
        rc_map_query = """
               select distinct
                    RTrim(sm.PremiseSrvClass) as RateClass,
                    RTrim(sm.LoadShapeTblSrvClass) as Map,
                    iif(t.TODQ is NULL, 0, iif(t.TODQ = 'Yes', 1, 0)) as TODQ
               from Premise as p
               full outer join CONED_TOD as t
                on Cast(t.TODCode as int) = Cast(p.RateClass as int)
               full outer join CONED_SClass_Map as sm
                on Cast(p.RateClass as int) = Cast(sm.PremiseSrvClass as int)
               where
                    sm.PremiseSrvClass is not NULL and
                    p.UtilityId = 'CONED'
               order by RateClass
                """

        df = pd.read_sql(rc_map_query, self.conn)
        df = df.applymap(lambda x: np.int(x))
        df.set_index('RateClass', inplace=True)
        return df

    def get_tod_map(self):
        tod_query = """
            select
                TODCode,
                iif(TODQ = 'Yes', 1, 0) as TODQ
            from CONED_TOD"""
        df = pd.read_sql(tod_query, self.conn)
        df = df.applymap(lambda x: np.int(x))
        df.set_index('TODCode', inplace=True)

        return df

    def get_coincident_peak(self):
        coincident_peak_query = """
            select
                cast(Year(CPDate) as varchar) as Year,
                CPDate,
                cast(HourEnding as int) as HourEnding
            from
                [CoincidentPeak]
            where
                UtilityID = 'CONED'
                """
        df = pd.read_sql(coincident_peak_query, self.conn)
        df.set_index('Year', inplace=True)
        return df

    def build_temperature_variant(self):
        """Conversion of raw weather station data into required
        temperature variant table. The temperature variant table
        is used to build the load shape adjustment table. The day and
        temperature value are utilized as indexing keys in the construction
        of the load shape adjustment table.

        Assigns the resulting table to:     self.temp_var
        """
        """
        THIS CODE IS DEPRECIATED AFTER CONED_NYWeatherData change on
        Feb 23, 2016

        # CONVERSIONS
        # convert hour into timedelta
        # increment `ObservedDate` by correspoding Timedelta
        # drop the `Hour` columns
        ts['Hour'] = ts['Hour'].apply(lambda x: pd.Timedelta(hours=x))
        ts['ObservedDate'] = ts['ObservedDate'] + ts['Hour']
        td = ts.drop('Hour', axis=1)

        # update index
        td.set_index('ObservedDate', inplace=True)

        # AGGREGATION
        # Station_i : sum = temperature + wetbulbtemperature
        td['RowSum'] = td.sum(axis=1)

        # drop columns aggregated in `RowSum`
        # pivot stations into columns
        # avg: (Station_i:RowSum) + (Station_j:RowSum) * (1/4)
        hr_avg = td.drop(['Temperature', 'WetBulbTemperature'], axis=1
                         ).pivot(columns='StationCode', values='RowSum'
                                 ).apply(lambda row: row.sum() * .25, axis=1)

        # ROLLING AVERAGE
        # group into days
        # create rolling window of 3 hours
        # average the window
        # take the maximum value per day
        ravg = hr_avg.groupby(pd.TimeGrouper('D')
                              ).rolling(window=3).mean().max(level=0)

        # ROLLING WEIGHTED SUM
        # applying weighted sum function
        def f(w):
            def g(x):
                return (w * x).sum()
            return g

        # required weights and rolling avg
        wts = np.array([.1, .2, .7])
        tv = ravg.rolling(window=3).apply(f(wts))
        """

        # QUERY
        # query and SQL execution
        temp_station_query = """select
                RTrim(StationCode) as StationCode,
                ObservedDate,
                Temperature, WetBulbTemperature
            from [CONED_NYWeatherData]
            order by
                ObservedDate"""

        ts = pd.read_sql(temp_station_query, self.conn)

        """
        Hourly Average:
            WetBulbTemperature = WBT; Temperature = T;

            for hour in ObservedDate:
                hourly_avg[i] = 0.25 * (KNYC_WBT + KNYC_T + KLGA_WBT + KLGA_T)

        """
        hourly_avg = pd.pivot_table(ts, index='ObservedDate',
                                    columns='StationCode',
                                    values=['Temperature',
                                            'WetBulbTemperature'],
                                    ).mean(1)

        """
        Rolling Average:
            1. Group hourly average into days
            2. Rolling mean over 3 hour window
            3. Take maximum average per day
        """
        daily_max_avg = hourly_avg.groupby(pd.TimeGrouper('D')
                                           ).rolling(window=3).mean().max(level=0)

        """
        Rolling Weighted Sum:
            The weighted sum is applyed to 3 day rolling window.
            The current day weight is 70%, day-1 is 20%, day-2 is 10%.

            weights = [0.1, 0.2, 0.7]
            day[i-2], day[i-1], day[i] = weights
        """
        # helper function to compute weighted sum
        def f(w):
            def g(x):
                return (w * x).sum()
            return g

        # Weights
        wts = np.array([.1, .2, .7])

        # Apply rolling weighted summation
        daily_max_avg.rolling(window=3).apply(f(wts))

        # CONVERSION
        # convert to DataFrame
        # reset the index
        temp_var = pd.DataFrame(daily_max_avg)
        temp_var.reset_index(inplace=True)
        temp_var.rename(columns={0: 'Max'}, inplace=True)

        # REMAP datetime.datetime to day_of_week; string
        # create day of week column and format
        days = {0: 'MON', 1: 'TUE', 2: 'WED',
                3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}

        # convert timestamp to day of week
        # convert numeric day to 3-letter identifier
        # convert datetime.datetime to string
        temp_var['DayOfWeek'] = temp_var['ObservedDate'].dt.dayofweek
        temp_var['DayOfWeek'] = temp_var['DayOfWeek'].apply(lambda x: days[x])
        temp_var.sort_values(by='ObservedDate', inplace=True)
        temp_var['ObservedDate'] = temp_var['ObservedDate'].apply(
            lambda x: x.strftime('%Y-%m-%d'))

        # slicing will be done on this table based on billing start/end dates.
        #   setting the index to 'ObservedDate' will simplify the process
        temp_var.set_index('ObservedDate', inplace=True)
        return temp_var

    def build_lst(self):
        """Imports the LoadShapeTable (LST). The LST is outer joined with the
        Temperature Variant table and indexed on the `ObservedDate`. This
        is an optimization that allows for a faster computation of the premise
        level LoadProfile
        """
        # Query; Execution
        lst_query = """select *
            from CONED_LoadShapeTempAdj
            where Strata != ''
            """
        lst = pd.read_sql(lst_query, self.conn)

        # Merging on Temperature Variants
        tmp = pd.merge(lst, self.temp_var.reset_index(),
                       left_on='DAYTYPE', right_on='DayOfWeek')

        # Filtering; lst.TEMP L <= tv.Max <= lst.TEMP U
        lower_mask = tmp['TEMP L BOUND'] <= tmp['Max']
        upper_mask = tmp['TEMP U BOUND'] >= tmp['Max']
        temperature_mask = (lower_mask == 1) & (upper_mask == 1)
        tmp = tmp.ix[temperature_mask]

        # Sort by ObservedDate and set index
        tmp.sort_values(by='ObservedDate', inplace=True)
        tmp.set_index('ObservedDate', inplace=True)

        return tmp

    def meter_logic(self, row):
        """Determines if a meter is VTOU based on TOD mapping.
        This distinction is utilized in the Scaling Trueup Factor
        selection.
        """
        tod_code, meter_type = row[['TOD', 'MeterType']]

        # Extract the tod mapping if it exists
        is_tod = 0  # assume False
        if tod_code in self.tod_map.index:
            is_tod = self.tod_map.ix[tod_code]['TODQ']

        if is_tod:
            return 'VTOU'
        return meter_type

    def is_tod(self, meter_logic):
        """Returns proper REGEX based on MeterLogic columns"""
        if meter_logic == 'VTOU':
            return 'T'
        return '[^T]'


class CONEDInterval(CONED):
    def __init__(self, conn=None):
        CONED.__init__(self, conn)
        self.hourly = pd.merge(self.get_hourly().reset_index(),
                               self.get_hourly_cp(),
                               how='left',
                               on=['PremiseId', 'Year']
                               ).set_index(['PremiseId', 'Year'])

        # split interval meters into VarinaceTrue; VarianceFalse
        # self.varTrue = self.hourly[self.hourly['VarTest'] == 1]
        # self.varFalse = self.hourly[self.hourly['VarTest'] == 0]

    def get_hourly_cp(self):
        hourly_cp_query = """
            select
                h.PremiseId,
                Year(h.UsageDate) as Year,
                h.Usage
            from [HourlyUsage] h
            inner join [CoincidentPeak] cp
                on cp.UtilityId = h.UtilityId
                and cp.CPDate = h.UsageDate
                and cp.HourEnding = h.HourEnding
            where h.UtilityId = 'CONED'"""

        return pd.read_sql(hourly_cp_query, self.conn)

    def get_hourly(self):
        """Retrieves all monthly profile record for each of the interval meters.
        Varinace Test:
            The sum of usage for the month containing the cp date has an
            absolute varinace less-equal to 4% when compared against billed
            usage.
                if ABS((billed_usage - sum) / billed_usage) ? 1 : 0;

        Returns:
            PremiseId, RateClass, ServiceClass, Zone, Stratum,
            TODCode, Year, BillStartDate, BillEndDate,
            BilledUsage, BilledDemand,
            Sum of usage for cp billing cycle,
            Variance Test result; 1 => True; 0 => False
        """

        hourly_query = """
            select
                h.PremiseId,
                p.RateClass,
                ce.[Service Classification],
                ce.[Zone Code] as ZoneCode,
                ce.[Stratum Variable] as Stratum,
                ce.[Time of Day Code] as TOD,
                Year(m.EndDate) as Year,
                m.StartDate,
                m.EndDate,
                m.Usage as BilledUsage,
                m.Demand as BilledDemand,
                Round(Sum(h.Usage), 0) as CPHourUsage,
                'INT' as MeterType,
                iif( Abs((m.Usage - (Sum(h.Usage))) / m.Usage) <= 0.04, 1, 0) as VarTest
            from
                [HourlyUsage] h
            inner join
                [MonthlyUsage] m
                on m.PremiseId = h.PremiseID
                and m.UtilityID = h.UtilityId
            inner join
                [CoincidentPeak] cp
                on cp.UtilityId = h.UtilityId
                and Year(cp.CPDate) = Year(m.EndDate)
                and (cp.CPDate between m.StartDate and m.EndDate)
                and (h.UsageDate between m.StartDate and m.EndDate)
            inner join
                [Premise] p
                on p.PremiseId = h.PremiseId
            inner join [ConED] ce
                on CAST(ce.[Account Number] as varchar) = h.PremiseId
            where
                (h.UtilityId = 'CONED') and
                (h.HourEnding between 1 and 24) and
                (cp.CPDate between p.EffectiveStartDate and p.EffectiveStopDate)
            group by
                h.PremiseId,
                p.RateClass, ce.[Service Classification],
                ce.[Zone Code], ce.[Stratum Variable], ce.[Time of Day Code],
                Year(m.EndDate),
                m.StartDate, m.EndDate,
                m.Usage,
                m.Demand
            having
                Count(h.Usage) = (DateDiff(hour, m.StartDate, m.EndDate) + 24)"""

        # obtain data; set defaults; converions
        df = pd.read_sql(hourly_query, self.conn)
        df['MCD'] = np.NaN
        df['NormUsage'] = np.NaN
        df['TOD'] = df['TOD'].apply(lambda x: np.int(x))

        # create multi-index; sort
        df.set_index(['PremiseId', 'Year'], inplace=True)
        df.sort_index(inplace=True)

        # determine meter type
        df['MeterLogic'] = df.apply(self.meter_logic, axis=1)
        df['MeterRegex'] = df['MeterLogic'].apply(tod_regex)

        return df

    def compute_mcd(self):
        if len(self.hourly['MCD'].unique()) == 1:
            self.hourly['MCD'] = self.hourly['Usage']

        varFalse = self.hourly[self.hourly['VarTest'] == 0]

        for rec in varFalse.itertuples():
            # Parse the record index
            prem, year = rec[0]

            # Parse record
            rate_class, strata, zone, stratum, tod, \
                bill_start, bill_end, usage, demand, sum_usage, \
                meter_type, var_test, mcd, normalized_usage, \
                meter_logic, meter_regex, cp_usage = rec[1:]

            # Convert rate_class to integer for proper Index value
            # Service class mapping
            rate_class = int(rate_class)
            # service_class = self.rc_map.ix[rate_class]['Map']

            # Slice billcycle from temperature variants
            billcycle = self.temp_var.ix[bill_start:bill_end]

            # Join bill cycle with LoadShapeAdjustmentTable (LST)
            local_lst = pd.merge(billcycle, self.lst,
                                 left_index=True, right_index=True,
                                 on=['Max', 'DayOfWeek'])

            # Filter for Straum condition
            stratum = float(stratum)
            stratum_lower_mask = local_lst['STRAT L BOUND'] <= stratum
            stratum_upper_mask = local_lst['STRAT U BOUND'] >= stratum
            stratum_mask = (stratum_lower_mask == 1) & (
                stratum_upper_mask == 1)
            local_lst = local_lst.ix[stratum_mask]

            # Filter for TimeOfDay meter type and Service Class Mapping
            tod_mask = local_lst['STRATA'].str.contains(meter_regex)
            sc_mask = (local_lst['SC'] == self.rc_map.ix[rate_class]['Map'])
            mask = (tod_mask == 1) & (sc_mask == 1)
            local_lst = local_lst.ix[mask]

            # Check for filtering condition:
            # local load shape table rows == billcycle rows
            if local_lst.shape[0] != billcycle.shape[0]:
                continue

            # Extract the kiloWatt hour columns
            kw_cols = [col for col in local_lst.columns if 'KW' in col]
            local_lst = local_lst[kw_cols]

            # Convert coincident peak information into usable keys
            # Compute the Customer Scaling Factor
            # Extract the Load Profile from the billing cycle
            # Compute the normalized usage
            cp_day, hr = self.cp.ix[str(year)]  # yyyy-mm-dd, hh
            csf = usage / local_lst.values.sum()
            load_profile = local_lst.ix[cp_day]['KW' + str(hr)]
            normalized_usage = load_profile * csf
            mcd = normalized_usage

            # Update the monthly usage values
            self.hourly.loc[(prem, year), ['NormUsage', 'MCD']
                            ] = [normalized_usage, mcd]

    def compute_icap(self):
        if len(self.hourly['MCD'].unique()) == 1:
            self.compute_mcd()

        def coned_icap(g):
            mcd = g['MCD'].values
            stf = g[g['ParameterId'] == 'SubzoneTrueupFactor']['Factor'].values
            ftf = g[g['ParameterId'] == 'ForecastTrueupFactor'][
                'Factor'].values

            try:
                icap = mcd[0] * stf[0] * ftf[0]
            except IndexError:
                icap = np.NaN
            return icap

        # Handle cases where Variance <= 4%

        # Get the CPDate/Hour Usage; Used as MCD
        """
        tmp = pd.merge(self.hourly.reset_index(), self.hourly_cp,
                       how='left',
                       on=['PremiseId', 'Year'])
        """

        # Join on the utility factors
        tmp = pd.merge(self.hourly.reset_index(), self.util,
                       how='left',
                       left_on=['Year', 'ZoneCode'],
                       right_on=['Year', 'Zone'])

        # Masking for proper variable selection; factor adjustment
        match_mask = tmp['MeterLogic'] == tmp['MeterType_y']
        all_mask = tmp['MeterType_y'] == 'ALL'
        mask = (match_mask == 1) | (all_mask == 1)

        tmp['Factor'] = tmp['Factor'].apply(lambda x: x + 1.0)

        labels = ['PremiseId', 'Year', 'RateClass', 'MeterLogic']
        icap_varTrue = tmp.ix[mask].groupby(labels
                                            ).apply(coned_icap).reset_index()

        icap_varTrue.rename(columns={0: 'ICap', 'MeterLogic': 'MeterType'},
                            inplace=True)

        icap_varTrue['Strata'] = icap_varTrue['RateClass']
        return meta_organize(self, icap_varTrue)


class CONEDMonthly(CONED):
    """CONEDMonthly implements the logic required to compute the ICap for both
    Demand and Consumption Meters.
    """

    def __init__(self, conn=None):
        CONED.__init__(self, conn)

        # loads all monthly records
        self.monthly = self.get_monthly()

    def get_monthly(self):
        """get_monthly returns Consumption and Demand meters. Value returned
        is a sorted dataframe with a MultiIndex.

        MultiIndex = [PremiseId, Year]
        """

        monthly_query = """select m.PremiseId,
                    p.RateClass, p.Strata,
                    ce.[Zone Code] as ZoneCode,
                    ce.[Stratum Variable] as Stratum,
                    ce.[Time of Day Code] as TOD,
                    Year(m.EndDate) as Year,
                    m.StartDate, m.EndDate,
                    m.Usage as BilledUsage,
                    m.Demand as BilledDemand,
                    iif(m.Demand is null, 'CON','DMD') as MeterType
                from [MonthlyUsage] m
                inner join [CoincidentPeak] cp
                    on cp.UtilityId = m.UtilityId
                    and Year(cp.CPDate) = Year(m.EndDate)
                    and (cp.CPDate between m.StartDate and m.EndDate)
                inner join [Premise] p
                    on p.PremiseId = m.PremiseId
                inner join [ConED] ce
                    on CAST(ce.[Account Number] as varchar) = m.PremiseId
                where
                    (m.UtilityId = 'CONED') and
                    (cp.CPDate between p.EffectiveStartDate and p.EffectiveStopDate) and
                    m.PremiseId not in (
                            select distinct PremiseId
                            from HourlyUsage
                            where UtilityId = 'CONED'
                    )"""

        # Execute query to obtain records
        # Initialize 'Metered Coincident Demand' (MCD) to NaN
        # Initialize 'NormalizedUsage' to NaN
        df = pd.read_sql(monthly_query, self.conn)
        df['MCD'] = np.nan
        df['NormUsage'] = np.nan

        # If BilledDemand == 0 then BilledDemand -> np.inf
        # Conversion is required for proper logic execution during
        # selection of MCD;
        # MCD = MIN(billed demand, normalized usage) where billed_demand > 0
        df['BilledDemand'] = df['BilledDemand'].replace(
            to_replace=0, value=np.inf)

        # Convert all TOD codes to integers. These values are used as
        # index locations to self.tod_map.ix. Indicies must be integers!
        df['TOD'] = df['TOD'].apply(lambda x: np.int(x))

        # Set a MultiIndex; sort by index
        df.set_index(['PremiseId', 'Year'], inplace=True)
        df.sort_index(inplace=True)

        # Determination if meter is VTOU or default meter
        df['MeterLogic'] = df.apply(self.meter_logic, axis=1)
        df['MeterRegex'] = df['MeterLogic'].apply(tod_regex)
        return df

    def compute_mcd(self):
        """compute_mcd is a mutator method. The 'Metered Coincident Demand'
        is calculated for each premise/year combination in the dataset.
        The resulting value is then stored in the original record set
        """

        for rec in self.monthly.itertuples():
            # Parse the record index
            prem, year = rec[0]

            # Parse record
            # Parse record
            rate_class, strata, zone, stratum, tod, \
                bill_start, bill_end, usage, demand, \
                meter_type, mcd, normalized_usage, \
                meter_logic, meter_regex = rec[1:]

            # Convert rate_class to integer for proper Index value
            # Service class mapping
            rate_class = int(rate_class)
            # service_class = self.rc_map.ix[rate_class]['Map']

            # Slice billcycle from temperature variants
            billcycle = self.temp_var.ix[bill_start:bill_end]

            # Join bill cycle with LoadShapeAdjustmentTable (LST)
            local_lst = pd.merge(billcycle, self.lst,
                                 left_index=True, right_index=True,
                                 on=['Max', 'DayOfWeek'])

            # Filter for Straum condition
            stratum = float(stratum) - 1.0
            stratum_lower_mask = local_lst['STRAT L BOUND'] <= stratum
            stratum_upper_mask = local_lst['STRAT U BOUND'] > stratum
            stratum_mask = (stratum_lower_mask == 1) & (
                stratum_upper_mask == 1)
            local_lst = local_lst.ix[stratum_mask]

            # Filter for TimeOfDay meter type and Service Class Mapping
            tod_mask = local_lst['STRATA'].str.contains(meter_regex)
            sc_mask = (local_lst['SC'] == self.rc_map.ix[rate_class]['Map'])
            mask = (tod_mask == 1) & (sc_mask == 1)
            local_lst = local_lst.ix[mask]

            # Check for filtering condition:
            # local load shape table rows == billcycle rows
            if local_lst.shape[0] != billcycle.shape[0]:
                self.monthly.loc[(prem, year), ['NormUsage', 'MCD']
                    ] = [False, False]
                continue

            # Extract the kiloWatt hour columns
            kw_cols = [col for col in local_lst.columns if 'KW' in col]
            local_lst = local_lst[kw_cols]

            # Convert coincident peak information into usable keys
            # Compute the Customer Scaling Factor
            # Extract the Load Profile from the billing cycle
            # Compute the normalized usage
            cp_day, hr = self.cp.ix[str(year)]  # yyyy-mm-dd, hh
            csf = usage / local_lst.values.sum()
            load_profile = local_lst.ix[cp_day]['KW' + str(hr)]
            normalized_usage = load_profile * csf
            mcd = np.min([normalized_usage, demand])

            # Update the monthly usage values
            self.monthly.loc[(prem, year), ['NormUsage', 'MCD']
                             ] = [normalized_usage, mcd]

    def compute_icap(self):

        # Mutatate monthly records
        # If the MCD has NOT been calculated then calculate MCD
        if self.monthly['MCD'].unique().shape[0] == 1:
            self.compute_mcd()

        print('self.monthly: ', self.monthly.columns)
        # Merge montly records with utility records
        # This merge makes the icap calculation simple using groupby
        tmp = pd.merge(self.monthly.reset_index(), self.util,
                       how='left',
                       left_on=['Year', 'ZoneCode'],
                       right_on=['Year', 'Zone'])

        print('tmp: ', tmp.columns)
        # All factors require adjustment by +1
        tmp['Factor'] = tmp['Factor'].apply(lambda x: x + 1.0)

        def coned_icap(g):
            """Computes the ICap value for a given Premise/Year combination.
            Arguments:
                g {group} -- grouped value with MCD, SubzoneTrueup,
                            ForecastTrueUp factors
            Returns:
                icap float|| np.nan
            """
            mcd = g['MCD'].values
            stf = g[g['ParameterId'] == 'SubzoneTrueupFactor']['Factor'].values
            ftf = g[g['ParameterId'] == 'ForecastTrueupFactor'][
                'Factor'].values
            try:
                icap = mcd[0] * stf[0] * ftf[0]
            except IndexError:
                icap = np.NaN
            return icap

        match_mask = tmp['MeterLogic'] == tmp['MeterType_y']
        all_mask = tmp['MeterType_y'] == 'ALL'
        mask = (match_mask == 1) | (all_mask == 1)

        labels = ['PremiseId', 'Year', 'RateClass', 'MeterLogic']
        icap = tmp.ix[mask].groupby(by=labels).apply(coned_icap).reset_index()

        # icap.drop('MeterType', axis=1, inplace=True)
        icap.rename(columns={0: 'ICap', 'MeterLogic': 'MeterType'},
                    inplace=True)

        # requires a Strata Column
        # icap['Strata'] = icap['RateClass']
        # icap['Strata'] = self.monthly['Strata']
        icap = pd.merge(icap, tmp, on=['PremiseId', 'Year', 'RateClass'])
        return meta_organize(self, icap)


class CONEDRecipe:
    def __init__(self, conn=None, results=None):
        if (conn is None) or (results is None):
            raise Exception('conn=%s, results=%s)' % (conn, results,))
        self.conn = conn
        self.Results = results

    def run_all(self):
        intv = CONEDInterval(self.conn).compute_icap()
        monthly = CONEDMonthly(self.conn).compute_icap()
        all_results = pd.concat([intv, monthly])
        res = self.Results(self.conn, all_results)
        return res


def meta_organize(obj_ref, df):
    """meta_organize updates the returned dataframe to include
    required information for database upload.

    The `params` dict is held in the CONED super class. Values
    are iterated over and assigned. Only the desired columns are
    returned.
    """
    keep = ['RunDate', 'ISO', 'Utility', 'PremiseId', 'Year',
            'RateClass', 'Strata', 'MeterType', 'ICap']

    # loop over params and update dataframe
    for k, v in obj_ref.params.items():
        df[k] = v

    # ICap years are adjusted by +1
    df['Year'] = df['Year'].apply(lambda yr: str(int(yr) + 1))

    return df[keep]


def tod_regex(meter_logic):
    """Returns proper REGEX based on rate class. Meters can be either
    TIME OF DAY (if TODQ == 1 then 'T') or
    NOT TIME OF DAY (if TODQ == 0 then '[^T]')
    """
    if meter_logic == 'VTOU':
        return 'T'
    return '[^T]'
