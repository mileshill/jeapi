#/home/ubuntu/miniconda3/envs/predictions/bin/python
from icap.database.icapdatabase import ICapDatabase
from icap.comed.comed import COMEDInterval


class C:
    def __init__(self):
        self.conn = ICapDatabase('./icap/database/icapdatabase.json').connect()
        self.c = COMEDInterval(self.conn)



if __name__ == '__main__':


    c = C().c
    # Testing for ACustCPL
    print('\nTesting ACustCPL')
    print('mean_pjm: Slicing PJM Mean CP Usage Values from self.pjm_cp_usage')
    mean_pjm = c.pjm_cp_usage[['PremiseId', 'Year', 'DSC', 'MeanPJM']].drop_duplicates().reset_index()
    print(mean_pjm.head())
    print('How many non "NaN" records"')
    print(mean_pjm.dropna(subset=['MeanPJM']).shape)

    c.pjm_cp_usage.to_csv('/tmp/pjm_cp_usage.csv')


    print(c.compute_icap().head())
