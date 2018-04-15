from icap.database.icapdatabase import ICapDatabase
from icap.ppl.ppl import PPLInterval

if __name__ == '__main__':
    conn = ICapDatabase('./icap/database/icapdatabase.json').connect()
    ppl = PPLInterval(conn)

    ppl.compute_icap()


