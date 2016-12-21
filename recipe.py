import numpy as np

class Recipe():
	'''
	Computes the ICap array for (utility, meter_type) 	
	'''

	def __init__(self, utility, meter_type, request):
		self.utility = utility
		self.meter_type = meter_type
		self.request = self.convert_values_to_int(request)
	
		# generated values	
		self.rows_ = None
		self.cols_ = None
		self.array_ = None	
	
	def convert_values_to_int(self, request):
		'''Convert values to correct numeric form.
		(min, max) = float or int
		step = int
		'''
		d = dict()
		for k,v in request.items():
			try:
				if 'step' in k:
					d[k] = int(v)
					continue
				d[k] = float(v)
			except ValueError:
				pass
			except KeyError:
				pass
		return d

	def compute_array(self):
		'''Selects the recipe from (utility, meter_type)'''
		utility = self.utility
		meter_type = self.meter_type
		if (utility, meter_type) == ('ppl', 'interval'):
			self.compute_ppl()
		else:
			pass

	def compute_ppl(self):
		'''
		Recipe approximation calculation for PPL:
		RECIPE (<avg usage> outer product <avg recon>) * rate class loss factor
		'''
		f = self.request
		# x-axis vector must be iterate in reverse for proper structure	
		usage_vec = np.linspace(f['usage_max'], f['usage_min'], num=80, dtype=np.float)
		recon_vec = np.linspace(f['recon_min'], f['recon_max'], num=40, dtype=np.float)[::-1]

		array = np.fliplr(np.outer(recon_vec, usage_vec))
		rows, cols = array.shape
		array_max = array.max()
		array_min = array.min()

		array_list = [[i, recon_vec[i], j, usage_vec[::-1][j] ,array[i,j]] for i in range(rows) 
						for j in range(cols)]
		self.rows_, self.cols_ = array.shape
		#self.array_ = array.tolist()
		self.array_ = array_list
		self.max_ = array_max
		self.min_ = array_min
		self.x_ = 'Average CP Usage'
		self.y_ = 'Average Reconcilation' 		
		
