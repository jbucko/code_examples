import numpy as np
import matplotlib.pyplot as plt

def draw_circle(x0,y0,rad):
	"""
	returns data of circle drawn around x0,y0 with radius rad
	"""
	angles = np.linspace(0,6.28,400)
	x = x0+rad*np.cos(angles)
	y = y0+rad*np.sin(angles)
	return [x,y]


