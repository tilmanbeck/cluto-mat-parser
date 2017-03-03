import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

class Mat2Tfidf_Parser:

	def __init__(self, mat_data_file):
		self._data_file = mat_data_file
		self._samples_count = 0
		self._features_count = 0
		self._tf_matrix = []
		self._tfidf_matrix = []

	def _tf(self, file_pointer):
		"""
		Parse file line by line and extract term counts for given positions
		""" 		
		for line in file_pointer:
			data = line.split()
			tmp = [0.0] * self._features_count
			for i in range(0, len(data), 2):
 				# indices in mat file are not zero-based
 				tmp[int(data[i])-1] = float(data[i+1])
			self._tf_matrix.append(tmp)

	def parse(self):
		# first line: samples_count word_count idk_count)
		f = open(self._data_file, 'r')
		header = f.readline().split()
		self._samples_count = int(header[0])
		self._features_count = int(header[1])
		self._tf(f)
		tfidf = TfidfTransformer(norm='l2')
		self._tfidf_matrix = tfidf.fit_transform(self._tf_matrix)

	def get_tfidf_matrix(self):
		"""
		Return tf-idf matrix of input. 
		"""
		return self._tfidf_matrix

	def get_tf_matrix(self):
		"""
		Return term-frequency matrix of input. 
		"""
		return self._tf_matrix

	def get_samples_count(self):
		return self._samples_count

	def get_features_count(self):
		return self._features_count

