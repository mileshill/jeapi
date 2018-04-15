"""
Title: concat.py

Purpose: concatenate all CSV files within a given list of files
"""
import pandas as pd
import numpy as np
import glob
from datetime import date


class FileConcatenate:
	timestamp = date.today().strftime('%Y_%m_%d')

	def __init__(self, *data_frames):
		self.data_frames = data_frames

	def __call__(self):
		df = pd.DataFrame()
		for frame in self.data_frames:
			df = df.append( frame.compare_ )
		#out_file = '_'.join([__class__.timestamp, 'recipe.csv'])
		#df.to_csv(out_file)
		df.to_csv('recipe.csv')				
