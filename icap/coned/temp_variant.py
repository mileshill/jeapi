import numpy as np
import pandas as pd


class Temperature_Variant:
    def __init__(self, query_result):
        self.query_result = query_result

    def build_temp_var():
    	ts = self.query_result
    	return pd.concat([self.temp_var_2014(ts),
                     self.temp_var_2015(ts),
                     self.temp_var_2016(ts)])


    def temp_var_2014(query_result):
        '''Compute temperature variant table for years <= 2014'''
        # helper to calculate rolling weights
        def f():
            w = np.array([.1, .2, .7])

            def g(x):
                return (w * x).sum()
            return g

        # y_i <= 2015
        # station_code == 'KNYC'
        year_mask = query_result.ObservedDate.apply(lambda d: d.year <= 2014)
        station_mask = query_result.StationCode == 'KNYC'

        df = query_result[(year_mask == 1) & (station_mask == 1)] \
            .drop(['Year', 'StationCode'], axis=1) \
            .set_index('ObservedDate') \
            .mean(axis=1) \
            .groupby(pd.TimeGrouper('D')) \
            .max() \
            .rolling(window=3) \
            .apply(f())

        return df

    def temp_var_2015(query_result):
        '''Compute temperature variant table for years == 2015'''
        year_mask = query_result.ObservedDate.apply(lambda d: d.year == 2015)
        hour_mask = query_result.ObservedDate.apply(
            lambda d: 9 <= d.hour <= 21)
        df = query_result[(year_mask == 1) & (hour_mask == 1)]

        """
	    Hourly Average:
	        WetBulbTemperature = WBT; Temperature = T;

	        for hour in ObservedDate:
	            hourly_avg[i] = 0.25 * (KNYC_WBT + KNYC_T + KLGA_WBT + KLGA_T)

	    """
        hourly_avg = pd.pivot_table(df, index='ObservedDate',
                                    columns='StationCode',
                                    values=['Temperature',
                                            'WetBulbTemperature']
                                    ).mean(1)
        """
	    Rolling Average:
	        1. Group hourly average into days
	        2. Rolling mean over 3 hour window
	        3. Take maximum average per day
	    """
        daily_max_avg = hourly_avg.groupby(pd.TimeGrouper('D')) \
            .rolling(window=3) \
            .mean() \
            .max(level=0)

        """
	    Rolling Weighted Sum:
	        The weighted sum is applyed to 3 day rolling window.
	        The current day weight is 70%, day-1 is 20%, day-2 is 10%.

	        weights = [0.1, 0.2, 0.7]
	        day[i-2], day[i-1], day[i] = weights 
	    """
        # helper function to compute weighted sum
        def f():
            w = np.array([.1, .2, .7])
            def g(x):
                return (w * x).sum()
            return g

        # Apply rolling weighted summation
        weighted_sum = daily_max_avg.rolling(window=3).apply(f())

        return weighted_sum


    def temp_var_2016(query_result):
	    def f():
	        w = np.array([.1, .2, .7])
	        def g(x):
	            return (w * x).sum()
	        return g
	    
	    year_mask = query_result.ObservedDate.apply(lambda d: d.year >= 2016)
	    df = query_result[year_mask]
	  
	    df = pd.pivot_table(df, index='ObservedDate', 
	                   columns='StationCode', 
	                   values=['Temperature', 'WetBulbTemperature']) \
	        .mean(axis=1) \
	        .groupby(pd.TimeGrouper('D')) \
	        .max() \
	        .rolling(window=3) \
	        .apply(f())
	    
	    return df
