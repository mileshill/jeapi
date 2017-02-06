import numpy as np
import pandas as pd


class Results():
    def __init__(self, conn, df):
        assert isinstance(df, pd.DataFrame)
        self.conn = conn
        self.df = df
        self.utilities = utility_transform(
            df['Utility'].drop_duplicates().values)
        self.compare_ = self.compare_historical().sort_values(
            by=['PremiseId', 'Year']).drop_duplicates()

        self.compare_.rename(
            columns={'Utility': 'UtilityId', 'ICap': 'RecipeICap',
                     'CapacityTagValue': 'HistoricalICap',
                     'HistVar': 'RecipeVariance'}, inplace=True)

    def compare_historical(self):
        """Compare historical computes the absolute variance between the
        recipe ICap and historical icap.
        """
        historical_query = """
            select
                PremiseId,
                Cast(CPYearID as varchar) as Year,
                CapacityTagValue
            from CapacityTagHistorical
            where UtilityId in {util}
        """.format(util=self.utilities)

        hist = pd.read_sql(historical_query, self.conn)
        compare = pd.merge(self.df, hist, on=['PremiseId', 'Year'], how='left')
        # compare.replace(to_replace=[0], value=[np.nan], inplace=True)
        actual = compare['CapacityTagValue']
        computed = compare['ICap']
        compare['RecipeVariance'] = abs((actual - computed) / actual) * 100.0

        return compare

    def write_comparison_to_csv(self, fp=None):
        """Write to CSV the result of an ICap computation and comparison
        against all historical icap values.
        """
        if fp is None:
            util = self.compare_['UtilityId'].drop_duplicates().values[0]
            fp = str(util).lower() + '_rec.csv'

        self.compare_.to_csv(fp,
                             na_rep='NULL',
                             float_format='%.4f',
                             index=False,
                             date_format='%Y-%m-%d_%H:%M'
                             )

    def analyze_comparison(self, write_to_excel=False):
        """
        Group by [metertype, year, rateclass, starta, valid] where valid
        has values:
            1 -> True
            0 -> False
            NULL

        Output is written to an xlsx file in the local directory.
        """
        # indicies of possible outcomes
        results = self.compare_.copy()

        # null_idx = results[pd.isnull(results['RecipeVariance'])].index
        # valid_idx = results[results['RecipeVariance'] <= 2.0].index
        # invalid_idx = results[results['RecipeVariance'] > 2.0].index

        # # assign values to outcomes on their index
        # results['Valid'] = ''
        # results.set_value(null_idx, 'Valid', 'NULL')
        # results.set_value(invalid_idx, 'Valid', 0)
        # results.set_value(valid_idx, 'Valid', 1)

        # # aggregate and count
        # details = results.groupby(['MeterType', 'Year', 'RateClass',
        #                            'Strata', 'Valid']
        #                      )[['RecipeICap', 'CapacityTagValue']].count()

        # utilizing pd.crosstab to provide better results
        def passing_value(series):
            variance = series['RecipeVariance']
            if variance <= 2.0:
                return True
            elif variance > 2.0:
                return False
            else:
                return 'NULL'

        results['Passing'] = results.apply(passing_value, axis=1)

        details = pd.crosstab([results['MeterType'],
                               results['RateClass'],
                               results['Strata'],
                               results['Year']],
                              results['Passing'], margins=True)

        # return the Dataframe
        if not write_to_excel:
            return pd.DataFrame(details)

        # write the dataframe and return results
        utility_name = results['UtilityId'].ix[0]
        file_name = utility_name + '_analysis.xlsx'
        try:
            writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
            output = pd.DataFrame(details)  # convert from multi-index series
            output.to_excel(writer, sheet_name=utility_name)
            writer.save()
            return output
        except Exception as e:
            return e


def utility_transform(utils):
    """Transform array of utilities into useable format
    for SQL query.

    INPUT: [A, B]
    OUTPUT: ('A', 'B')
    """
    def utility_formatter(util_string):
        return "'" + str(util_string) + "'"

    return "(" + ", ".join(map(utility_formatter, utils)) + ")"
