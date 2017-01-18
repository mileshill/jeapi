#!flask/bin/python
import json
from recipe import Recipe
from flask import Flask
from flask import jsonify, make_response, request

# App instance
api_app = Flask(__name__)

# Mongo Client
#from flask_pymongo import PyMongo

from pymongo import MongoClient
#mongo = PyMongo(api_app)
client = MongoClient('localhost', 27017)
db = client.ppl
col = db.recipe
'''
with api_app.app_context():
    print(dir(mongo.db))
    print(mongo.db.recipe.find_one_or_404({'UtilityId':'PPL'}))
    '''

# Handle cross-domain errors
from flask_cors import CORS, cross_origin
CORS(api_app)

############################## Globals ############################## 
class Global:
    ''' Container for persistent variables '''
    def __init__(self):
        pass
    def change():
        pass
def validate_route_keys(g, endpoint_input):
    '''Confirms endpont_input matches they keys
    in recipe.json.

    INPUT: Global object, iterable with (iso, utility, meter_type)
    RETURN: (Boolean, dict)
    '''
    assert isinstance(g, Global)

    # load and return the recipe parameters
    try:
        iso, utility, meter_type = endpoint_input
        recipe = g.recipe_dict[iso][utility][meter_type]
        return True, recipe
    except KeyError as invalid_key:
        return False, {'invalid_key':invalid_key[0]}

############################## Preprocessing ############################## 

# Load the recipe JSON and store as global.
# JSON should be present for all recipe calls.
g = Global()

try:
    with open('recipe.json') as f:
        g.recipe_dict = json.load(f)
except FileNotFoundError:
    g.recipe_dict = None
    ############################## Routing ############################## 
    @api_app.route('/')
    def index():
        ''' Hello world statement '''
        return 'Hello World'


#-- Recipe Population
@api_app.route('/populate/<iso>/<utility>/<meter_type>', methods=['GET'])
#@crossdomain(origin='*')
def populate_recipe(iso, utility, meter_type):
    '''Populate recipe accepts the ISO/utility/meter_type combination
    and returns the necessary information to build UI widgets.

    INPUT: iso, utility, meter_type
    OUTPUT: JSON for webform population

    '''
    # route validation
    endpoint_input = (x.lower() for x in (iso, utility, meter_type))	
    valid_recipe, recipe_params = validate_route_keys(g, endpoint_input)
    if valid_recipe:
        return make_response(jsonify(recipe_params), 200)
    return error_bad_input(recipe_params)	

#-- Compute Sensitivity
@api_app.route('/compute/<iso>/<utility>/<meter_type>', methods=['POST'])
#@crossdomain(origin='*')
def compute_recipe(iso, utility, meter_type):
    '''Compute the sensitivity array for given input params.
    INPUT: Consumes JSON with params
    RETURN: Array of ICap values
    '''
    # route validation
    endpoint_input = (x.lower() for x in (iso, utility, meter_type))
    valid_recipe, recipe_params = validate_route_keys(g, endpoint_input)		
    if not valid_recipe:
        return error_bad_input(recipe_params)

    # recipe param validation
    recipe_keys = recipe_params['params'].keys()
    post_keys = [k.split('_')[0] for k in request.form.keys()]
    diff = set(recipe_keys) - set(post_keys)

    # are all recipe keys represented in the post?
    if diff:
        missing = {'missing_keys':list(diff)}
        return error_bad_input(missing)

    # Time to compute the ICAP
    # all elements are present, create the array
    # for PPL:
        #	<usage> * <recon> * rclf

    recipe = Recipe(utility, meter_type, request.form)
    recipe.compute_array()

    results = {'array': recipe.array_,
               'rows': recipe.rows_,
               'cols': recipe.cols_,
               'max': recipe.max_,
               'min': recipe.min_,
               'x_label':recipe.x_,
               'y_label':recipe.y_}

    return make_response(jsonify(results), 200)

import numpy as np
import pandas as pd
@api_app.route('/gethistory/<iso>/<utility>/<premise_id>', methods=['GET'])
def get_history(iso, utility, premise_id):
    params = {'Utility':utility.upper(), 'PremiseId':long(premise_id)}
    results = [r for r in col.find(params, {'_id':False})]

    if not results:
        return make_response(jsonify({}), 404)	

    df = pd.DataFrame(list(results))
    df['PercentChange'] = df.RecipeICap.pct_change()
    df['MAD'] = df['RecipeICap'].mad()
    out = df.to_dict(orient='records')	


    return make_response(jsonify(out), 200)	

from icap.database.icapdatabase import ICapDatabase
from icap.correlation.correlation import Correlation
@api_app.route('/correlation/<year>/<iso>/<utility>/<premise>', methods=['GET'])
def correlation_analysis(year, iso, utility, premise):
    fp = 'icap/database/icapdatabase.json'
    conn = ICapDatabase(fp).connect()

    corr = Correlation(conn, year=year, iso=iso, utility=utility, premise=premise)
    r = corr.analyze()
    return make_response(jsonify(r.results()), 200)

############################## Error Handling ############################## 
@api_app.errorhandler(404)
def error_not_found(error):
    '''General 'Not Found' error code'''
    return make_response(jsonify({'error':'Not Found'}), 404)


@api_app.errorhandler(400)
def error_bad_input(bad_input):
    '''Bad input to endpoint'''
    return make_response(jsonify({'error':bad_input}), 400)

############################## Launch api_app ############################## 
if __name__ == '__main__':
    host = '0.0.0.0'
    port = 3000
    debug = True 
    threaded=True

    api_app.run(debug=debug, host=host, port=port, threaded=threaded)
