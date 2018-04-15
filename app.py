#!/home/ubuntu/miniconda3/envs/predictions/bin/python
import os

fp = os.path.dirname(os.path.realpath(__file__))
os.chdir(os.path.realpath(fp)) # change into this directory for execution

import subprocess

from flask import Flask
from flask import jsonify, make_response, Response
from flask_cors import CORS, cross_origin
from icap.correlation.correlation import Correlation, CorrelationException
from icap.database.icapdatabase import ICapDatabase

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



@app.route('/')
@cross_origin('*')
def index():
    return 'App is running'


@app.route('/launchrecipe')
@cross_origin('*')
def launch_recipes():
    subprocess.call('nohup run_recipes.sh &', shell=True)
    return 'Recipe calculations started'


@app.route('/correlation/<iso>/<utility>/<premise_id>', methods=['GET'])
@cross_origin('*')
def correlation(iso, utility, premise_id):
    # convert to lowercase
    iso, utility, premise_id = [str(x).lower() for x in [iso, utility, premise_id]]
	
    # update pjm for database standard
    if '-rto' in iso:
        iso = iso.split('-')[0]

    # validate input
    if not valid_iso_util(iso=iso, utility=utility):
        return make_response("InvalidCombination: {0} : {1}".format(iso, utility)), 404

    # connect to database
    with make_connection() as conn:
        try:
            if iso == 'pjm':
                iso += '-rto'
            corr_result = Correlation(conn=conn, iso=iso.upper(), utility=utility.upper(), premise=premise_id) \
                .analyze() \
                .results()
        except CorrelationException as e:
            print(e.message)
            return make_response(e.message), 404

        return make_response(jsonify(corr_result)), 200


def make_connection():
    file_path = os.path.join(os.curdir, 'icap', 'database', 'icapdatabase.json')
    return ICapDatabase(file_path).connect()


def valid_iso_util(iso, utility):
    valid_combinations = {'pjm': ['ppl', 'peco', 'pseg', 'comed'], 'nyiso': ['coned', 'centhud']}
    if (valid_combinations.get(iso, False)) and (utility in valid_combinations[iso]):
        return True
    return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1337, debug=True)
