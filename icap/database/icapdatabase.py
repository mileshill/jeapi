import pymssql
import json



class InstanceException(Exception):
    '''Raises error of object improperly initialized'''
    def __init__(self, message):
        self.message = message

class ICapDatabase():
    '''Handles connection to AWS Database'''
    def __init__(self, fp=None):

        # requires filepath for config file
        if not fp:
            raise InstanceException('Filepath to database config required')

        # config file must be json
        if not fp[-4:] == 'json':
            raise InstanceException('Database config must be json')

        # load config file
        with open(fp, 'r') as f:
            db = json.load(f)

        # validate required keys present
        required_keys = {'ip', 'user', 'password', 'port'}
        db_keys = set(db.keys())
        assert len(required_keys.intersection(db_keys)) == len(required_keys)

        # assign connection protocol
        self.server = db['ip']
        self.user = db['user']
        self.password = db['password']
        self.port = int(db['port'])

    def connect(self):
        '''Returns connection object'''
        conn = pymssql.connect(
                server = self.server,
                user = self.user,
                password = self.password,
                port = self.port)

        assert isinstance(conn, pymssql.Connection)
        return conn


