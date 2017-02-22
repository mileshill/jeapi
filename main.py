# Connection; Results
from icap.database.icapdatabase import ICapDatabase
from icap.results.results import Results

# Recipes
from icap.pseg.pseg import PSEGRecipe
from icap.peco.peco import PECORecipe
from icap.ppl.ppl import PPLRecipe
from icap.coned.coned import CONEDRecipe
from icap.centhud.centhud import CENTHUDRecipe

# Libraries
# import pandas as pd
from datetime import datetime

def main():
    fp = 'icap/database/icapdatabase.json'
    conn = ICapDatabase(fp).connect()

    # Compute recipes for all meter types and return historical comparison
    # Returned value is a `Results` object

    peco = PECORecipe(conn, Results).run_all()
    #centhud = CENTHUDRecipe(conn, Results).run_all()
    pseg = PSEGRecipe(conn, Results).run_all()
    ppl = PPLRecipe(conn, Results).run_all()
    coned = CONEDRecipe(conn, Results).run_all()

    # *.write_to_csv() writes the records
    # *.analyze_comparison() groups into multiindex and outputs
    #   values grouped on [metertype, year, rateclass, strata] with
    #   possibles fields of
    #       Field -> Value
    #       True  -> 1
    #       False -> 0
    #       NaN   -> NULL

    peco.write_comparison_to_csv()
    peco.analyze_comparison(write_to_excel=True)

    #centhud.write_comparison_to_csv()
    #centhud.analyze_comparison(write_to_excel=True)

    pseg.write_comparison_to_csv()
    pseg.analyze_comparison(write_to_excel=True)

    ppl.write_comparison_to_csv()
    ppl.analyze_comparison(write_to_excel=True)

    coned.write_comparison_to_csv()
    coned.analyze_comparison(write_to_excel=True)


if __name__ == '__main__':
    main()
