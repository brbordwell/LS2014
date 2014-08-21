import os
import numpy as np
import pylab as pl

"""
Small script for testing visualization
"""

file_path = os.path.join(os.path.dirname(__file__), os.pardir, 'Data', 'GalaxyZooTraining_CherenkovRad.csv')
data = np.genfromtxt(file_path, delimiter=',', names=True)
eliptical, spiral, u, g, i, z = data['p_el_debiased'] > .8, data['p_cs_debiased'] > .8, data['u'], data['g'], data['i'], data['z']
unknown = np.ones(len(eliptical), dtype=bool) & (~eliptical) & (~spiral)
colors = u - i, i
names = np.array(["u - i", "i"])
cl_names = ["eliptical", 'spiral', 'unknown']
pl.figure(figsize = (10, 10))
pl.plot(colors[0][eliptical], colors[1][eliptical], "o", markersize=5, label = cl_names[0])
pl.plot(colors[0][spiral], colors[1][spiral], "o", markersize=5, label = cl_names[1])
pl.plot(colors[0][unknown], colors[1][unknown], "o", markersize=5, label = cl_names[2])
pl.xlabel(names[0])
pl.ylabel(names[1])
pl.legend(loc="best")
pl.show()