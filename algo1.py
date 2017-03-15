# Mohammed Ajmal 

# Method 1 : using Gibb's method and averaging to find the orbit needed.

"""
1. Parse the file and get (x,y,z) set of coordinates.
2. Choose random points (set of 3) from the data set.
3. Find orbital parameters.
4. Repeat above set for given number of iterations.
5. Output the averaged value of orbital parameters.
"""
# physical constants
mu = 398600.0


import matplotlib.pylab as plt
import numpy as np
from numpy.linalg import norm
from random import randint
import math

def readXYZ(filename):
	x = []; y = [];	z = []

	file = open(filename, "r")
	str = file.readlines()

	for line in str:
		parts = line.split('\t')
		try : 
			#for errorless datas
			#x.append(float(parts[1]))
			#y.append(float(parts[2]))
			#z.append(float(parts[3]))
			#for jittery data 
			x.append(float(parts[4]))
			y.append(float(parts[5]))
			z.append(float(parts[6]))
		except ValueError:
			print ''

	file.close();
	return (x,y,z)

def get3Coordinates(x, y, z):
	# We select the set of 'a' coordinates into group of three and 
	# select a coordinate randomly from each group. 
	# This is done to ensure atleast some level of spacing between
	# three points selected.
	a = len(x)
	i  = randint(0, a/3)
	j  = randint(a/3, 2*a/3)
	k  = randint(2*a/3, a-1)

	if i==j | j==k | k==i :
		raise ValueError('Randomly chosen points happened to be same')
	print  (i,j,k ) 
	return ( (x[i], y[i], z[i]), (x[j], y[j], z[j]) ,(x[k], y[k], z[k]) ) 


def getOrbitPlaneDirection(a,b,c):

	# there are two possible directions for any plane.
	# say, if one is x, other one should be -x .
	# This function always returns the direction with positive value
	# of z- coordinate, just as a convention
	# (x is a unit vector)

	x = np.cross (np.array(a)-np.array(b), np.array( c) -np.array(b) )
	mod = norm(x)

	# returns the unit vector with positive z component
	if  x[2]/mod<0:
		return (-1*x[0]/mod, -1*x[1]/mod, -1*x[2]/mod)
	else : 
		return (x[0]/mod, x[1]/mod, x[2]/mod)

def getInclination(h):
	# h is the direction of orbital plane with respect to ECI frame
	return (math.acos(h[2] /norm(h)) * 180.0 / math.pi)

def getRaan(h):
	# h is the direction of orbital plane with respect to ECI frame
	n = np.cross(np.array([0,0,1]), np.array(h))
	return (math.acos(n[0] /norm(n)) * 180.0 / math.pi)

def getVelocity(a, b, c):
	# Gibb's method to get velocity of three points, which we feed in order
	# Algorithm: 
	# 1) Get position vectors a, b, c
	# 2) Calculate aXb, bXc, cXa
	# 3) Verify that a.(bXc) =0
	# 4) Calculate N, D and S
	# 5) Calculate va, vb, vc
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)
	moda = norm(a)
	modb = norm(b)
	modc = norm(c)
	aXb = np.cross(a,b)
	bXc = np.cross(b,c)
	cXa = np.cross(c,a)

	if abs(np.dot( a, aXb)) > 0.00001:
		raise ValueError('Vectors are not coplanar')
   	
	N = aXb*modc + bXc*moda + cXa*modb
	D = aXb + bXc + cXa
	S = a*(modb - modc) + b*(modc - moda) + c*(moda - modb)

	va = ((mu/(norm(N)* norm(D)))**0.5) * (S + (np.cross (D,a)/moda)) 
	vb = ((mu/(norm(N)* norm(D)))**0.5) * (S + (np.cross (D,b)/modb))
	vc = ((mu/(norm(N)* norm(D)))**0.5) * (S + (np.cross (D,c)/modc)) 

	return (va, vb, vc)

def getE(r,v):
	r = np.array(r)
	v = np.array(v)
	modr = norm(r)
	modv = norm(v)
	e = ((modv**2 - (mu/modr))*r  - (np.dot(r,v))*v )/mu
	return e

def getPeriapsisArgument(h,e):
	n = np.cross(np.array([0,0,1]), np.array(h))
	return math.acos(  np.dot(n,e) / (norm(n)* norm(e))) * 180.0 / math.pi

def getMajorAxis(h,e):

	h = norm(h)
	e = norm(e)
	apo = (h**2 / (mu*(1-e)))
	peri = (h**2 / (mu*(1+e)))
	return (abs(apo)+abs(peri))/2

def findOrbitalParams(x,y,z, itr):

	sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
	v = 0
	while (v <itr):
		try:
			a, b, c = get3Coordinates(x,y,z) # returns lists a,b,c
			planeDir = getOrbitPlaneDirection(a,b,c)
			i 	= getInclination(planeDir)
			raan 	= getRaan(planeDir)
			va, vb, vc = getVelocity(a,b,c) # returns array
			e 	= getE(a, va) # returns array
			abse	= norm(e)
			h 	= np.cross(a, va)
			majora 	= getMajorAxis(h,e)
			omega  	= getPeriapsisArgument(planeDir,e)

			sum	+= np.array([majora, abse, i, raan, omega])
			#print (majora, abse, i, raan, omega)
			v 	+= 1
		except ValueError as err:
			print ''

	#print v
	return sum/(v)

def plotValue(values):
	plt.plot(values,   "*")
	plt.grid()
	plt.show()	

if __name__ == '__main__':
	
	iterations = 10
	x, y, z =  readXYZ("trackdata.csv");	
	params = findOrbitalParams(x,y,z, iterations)
	
	print '( a, e, i, raan, omega ) \n', params
