#!flask/bin/python

# General 
import os
import unittest
import json

# Coverage loading and initialization 
from coverage import coverage
cov = coverage(branch=True, omit=['flask/*', 'justenergy_tests.py', 'icap/*'])
cov.start()

# App
from justenergy_api import *

# Testing Framework
class JustEnergyTestCase(unittest.TestCase):
	
	def setUp(self):
		self.app = api_app.test_client()
		self.app.testing = True

		self.post_payload_VALID =  dict(
			rclf='1.0', 
			recon_min=1.019, recon_max=1.200, recon_steps=40,
			usage_min=800, usage_max=900, usage_steps=20)
	
		self.post_payload_INVALID =  dict(
			rclf='1.0', 
			recon_min=1.019, recon_max=1.200, recon_steps=40)



		# Global object and recipe loading
		g = Global()
		try:
			with open('recipe.json','r') as f:
				g.recipe_dict = json.load(f)
		except FileNotFoundError:
			g.recipe_dict = None
		
	
	def tearDown(self):
		pass
	#################### Preprocessing #################### 	
	def test_global_load_recipe(self):
		self.assertIsNotNone(g.recipe_dict)	

	#################### Index #################### 	
	def test_home_status_code(self):
		# Check status
		result = self.app.get('/')
		self.assertEqual(result.status_code, 200)

	def test_index_return(self):
		result = self.app.get('/')
		self.assertEqual(result.data, 'Hello World')

	#################### Validate Recipe Keys ########## 	
	def test_validate_route_keys_VALID(self):
		endpoint_input = ('pjm', 'ppl', 'interval')
		valid, _ = validate_route_keys(g, endpoint_input)	
		self.assertTrue(valid)

	def test_validate_route_keys_INVALID(self):
		endpoint_input = ('pjm', 'p', 'in')
		valid, _ = validate_route_keys(g, endpoint_input)
		self.assertFalse(valid)

	#################### Populate #################### 	
	def test_populate_recipe_status_code(self):
		result = self.app.get('/populate/pjm/ppl/interval')
		self.assertEqual(result.status_code, 200)

	def test_populate_recipe_POST(self):
		result = self.app.post('/populate/pjm/ppl/interval')
		# 405 -> Method Not Allowed
		self.assertEqual(result.status_code, 405)

	def test_populate_recipe_valid(self):
		result = self.app.get('/populate/pjm/ppl/interval')
		result_d = json.loads(result.data)
		
		self.assertIsInstance(result_d, dict)	
		
	def test_populate_recipe_invalid(self):
		result = self.app.get('/populate/pjm/ppl/2')
		self.assertEqual(result.status_code, 400)


	#################### Compute  #################### 	
	def test_compute_recipe_invalid_method(self):
		'''GET is not allowed method'''
		result = self.app.get('/compute/pjm/ppl/interval')
		self.assertEqual(result.status_code, 405)

	def test_compute_recipe_invalid_params(self):
		result = self.app.post('/compute/pjm/pp/in', {})
		self.assertEqual(result.status_code, 400)

	def test_compute_recipe_valid_params(self):
		result = self.app.post('/compute/pjm/ppl/interval', 
			data=self.post_payload_VALID,
			follow_redirects=True)
		self.assertEqual(result.status_code, 200)

	#################### Errors  #################### 	
	def test_error_404(self):
		result = self.app.get('/populate/null') 
		self.assertEqual(result.status_code, 404)

class PSEGTestCase(unittest.TestCase):

	def setUp(self):
		from icap.database.icapdatabase import ICapDatabase
		self.conn = ICapDatabase().connect()

	def test_connection(self):
		self.assertIsInstance(conn, pymssql.Connection)



# Evaluation
if __name__ == '__main__':
	try:
		unittest.main()
	except:
		pass
	
	cov.stop()
	cov.save()
	print('\n\nCoverage Report:\n')
	cov.report()
	cov.html_report(directory='/home/ubuntu/just_energy/coverage')	
	cov.erase()
