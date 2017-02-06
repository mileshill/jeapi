#!/usr/bin/env python3.5
import unittest
import numpy as np

# DATABASE CONNECTION and RESULTS
from icap.database.icapdatabase import ICapDatabase
from icap.results.results import Results

###############################################################################
#
# PECO
#
###############################################################################

from icap.peco.peco import PECODemand
@unittest.skip('Cascading failure due to usage difference in test case and database')
class PECODemandTestCase(unittest.TestCase):

    def setUp(self):
        fp = 'icap/database/icapdatabase.json'
        self.conn = ICapDatabase(fp).connect()
        self.premise = '3612801108'
        self.year = '2015'
        self.icap_year = '2016'
        self.rate_class = 'GS'
        self.strata = '101'

        # numeric comparisions
        self.places = 3

        # load the testcase
        self.test_case = PECODemand(self.conn, premise=self.premise)

    def test_2015_demand_values(self):
        """Compare demand values in testcase vs database
        """
        # test case premise
        dmd = self.test_case

        # filtering for values
        dmd.records_.sort_values(by='StartDate', inplace=True)

        # values to test
        db_demand = dmd.records_[dmd.records_['Year'] == self.year]['Demand'].values
        test_case_demand = np.asarray([15.2, 15.2, 14.4, 15.2], dtype=np.float)


        # all close
        self.assertTrue(
                np.testing.assert_allclose(
                    test_case_demand, db_demand,
                    rtol=1e-03))


    def test_2015_avg_demand(self):
        dmd = self.test_case

        # filtering for values
        dmd = dmd.records_[dmd.records_['Year'] == self.year]

        # values to test
        db_avg_dmd = dmd['Demand'].mean()
        test_case_avg_dmd = 15.0

        # almost equal
        self.assertAlmostEqual(test_case_avg_dmd, db_avg_dmd, places=self.places)


    def test_2015_weather_correction_factor(self):

        util = self.test_case.util_df_

        # filtering for values
        year_mask = util['Year'] == self.year
        rc_mask = util['RateClass'] == self.rate_class
        strata_mask = util['Strata'] == self.strata
        naratio_mask = util['ParameterId'] == 'NARatio'

        mask = (rc_mask == 1) & (strata_mask == 1) & \
                (naratio_mask == 1) & (year_mask == 1)

        # values to test
        db_wcf = util[mask].mean().iloc[0]
        test_case_wcf = 0.93

        # almost equal
        self.assertAlmostEqual(test_case_wcf, db_wcf, places=self.places)


    def test_2015_rateclass_loss_factor(self):
        util = self.test_case.util_df_

        # filtering for values
        year_mask = util['Year'] == self.year
        rc_mask = util['RateClass'] == self.rate_class
        strata_mask = util['Strata'] == self.strata
        rc_mask = util['ParameterId'] == 'RateClassLoss'

        mask = (rc_mask == 1) & (strata_mask == 1) & \
                (rc_mask == 1) & (year_mask == 1)

        # values to test
        db_rclf = util[mask]['ParameterValue'].iloc[0]
        test_case_rclf = 1.1031

        # almost equal
        self.assertAlmostEqual(test_case_rclf, db_rclf, places=self.places)


    def test_2015_PLC_factor(self):
        sys = self.test_case.sys_df_

        # filtering for values
        year_mask = sys['Year'] == self.year
        rc_mask = sys['ParameterId'] == 'PLCScaleFactor'
        mask = (year_mask == 1) & (rc_mask == 1)

        # values to test
        db_plc = sys[mask]['ParameterValue'].iloc[0]
        test_case_plc = 0.9766

        # almost equal
        self.assertAlmostEqual(test_case_plc, db_plc, places=self.places)

    def test_2016_icap(self):
        dmd = self.test_case
        icap = dmd.compute_icap()

        # compare icap to historic values
        r = Results(self.conn, icap).compare_historical()

        # verify the variance
        variance = r['HistVar'].iloc[0]
        self.assertLessEqual(variance, 2.0)


from icap.peco.peco import PECOConsumption
@unittest.skip('No records for premise=5811500607')
class PECOConsumptionTestCase(unittest.TestCase):
    """Compare known values in TestCase"""
    pass

from icap.peco.peco import PECOInterval
class PECOIntervalTestCase(unittest.TestCase):
    def setUp(self):
        fp = 'icap/database/icapdatabase.json'
        self.conn = ICapDatabase(fp).connect()
        self.premise = '8722801404'
        self.year = '2014'
        self.icap_year = '2015'
        self.rate_class = 'HT'
        self.strata = '151'

        # numeric comparisions
        self.places = 3

        # load the testcase
        self.test_case = PECOInterval(self.conn, premise=self.premise)

    def test_2014_usage(self):
        rec = self.test_case.records_

        db_usage = rec[rec['Year'] == self.year
                ].sort_values(by='UsageDate')['Usage'].values

        test_case_usage = np.asarray([740.63, 764.63, 797.25, 785.25, 863.63])


        # if assertion is True then None is returned
        self.assertIsNone(
                np.testing.assert_allclose(
                    test_case_usage, db_usage,
                    rtol=1e-03))


    def test_2015_icap(self):
        # compute the icap
        icap = self.test_case.compute_icap()

        # obtain comparison dataframe
        compare = Results(self.conn, icap).compare_historical()

        # extract the historical variance
        variance = compare[compare['Year'] == self.icap_year]['HistVar'].iloc[0]

        # less than equal to
        self.assertLessEqual(variance, 2.0)

###############################################################################
#
# PSEG
#
###############################################################################
"""These tests are redundant. Should they be consolidated?"""

from icap.pseg.pseg import PSEGDemand
class PSEGDemandTestCase(unittest.TestCase):
    def setUp(self):
        fp = 'icap/database/icapdatabase.json'
        self.conn = ICapDatabase(fp).connect()

        self.premise = 'PE000007932336623641'
        self.icap_year = '2016'
        self.test_case = PSEGDemand(self.conn, premise=self.premise)

    def test_2016_varinace(self):
        recipe_output = self.test_case.compute_icap()
        compare = Results(self.conn, recipe_output).compare_historical()

        variance = compare[compare['Year'] == self.icap_year]['HistVar'].iloc[0]
        self.assertLessEqual(variance, 2.0)

from icap.pseg.pseg import PSEGConsumption
class PSEGConsumptionTestCase(unittest.TestCase):
    def setUp(self):
        fp = 'icap/database/icapdatabase.json'
        self.conn = ICapDatabase(fp).connect()

        self.premise = 'PE000010659852776129'
        self.icap_year = '2016'
        self.test_case = PSEGConsumption(self.conn, premise=self.premise)

    def test_2016_varinace(self):
        recipe_output = self.test_case.compute_icap()
        compare = Results(self.conn, recipe_output).compare_historical()

        variance = compare[compare['Year'] == self.icap_year]['HistVar'].iloc[0]
        self.assertLessEqual(variance, 2.0)


from icap.pseg.pseg import PSEGInterval
class PSEGIntervalTestCase(unittest.TestCase):
    def setUp(self):
        fp = 'icap/database/icapdatabase.json'
        self.conn = ICapDatabase(fp).connect()

        self.premise = 'PE000011920969080047'
        self.icap_year = '2016'
        self.test_case = PSEGInterval(self.conn, premise=self.premise)

    def test_2016_varinace(self):
        recipe_output = self.test_case.compute_icap()
        compare = Results(self.conn, recipe_output).compare_historical()

        variance = compare[compare['Year'] == self.icap_year]['HistVar'].iloc[0]
        self.assertLessEqual(variance, 2.0)

###############################################################################
#
# PPL
#
###############################################################################
from icap.ppl.ppl import PPLInterval
class PPLIntervalTestCase(unittest.TestCase):
    def setUp(self):
        fp = 'icap/database/icapdatabase.json'
        self.conn = ICapDatabase(fp).connect()

        self.premise = '6313091035'
        self.icap_year = '2016'
        self.test_case = PPLInterval(self.conn, premise=self.premise)

    def test_2016_variance(self):
        recipe_output = self.test_case.compute_icap()
        compare = Results(self.conn, recipe_output).compare_historical()

        variance = compare[compare['Year'] == self.icap_year]['HistVar'].iloc[0]
        self.assertLessEqual(variance, 2.0)



if __name__ == '__main__':
    unittest.main()
