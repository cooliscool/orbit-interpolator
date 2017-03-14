# Algorithm 

# 1) parse x, y, z coordinates from file.
# 2) fit the curve for obtained x, y coordinate set (an ellipse in XY plane)
# 3) choose random 3 (x,y,z) points from the dataset and find direction of orbital plane 
#    Equation of a plane can be found using any three points on the plane. 
# 4) do the same for several combinations of 3 point set (for getting an average estimate)
# 5) rotate the plane to the target plane coordinate system, which is the final orbit 
# 6) calculate the keplerian orbital elements (i,raan, argument_of_periapsis, a, e)

# Method 1 : using Gibb's method and averaging to find the orbit needed.


# physical constants
mu = 398600.0


import matplotlib.pylab as plt
import numpy as np
from numpy.linalg import eig, inv
from random import randint
import math

def readXYZ(filename):
	x = []
	y = []
	z = []

	file = open(filename, "r")

	str = file.readlines()

	for line in str:
		parts = line.split('\t')
		try : 
			x.append(float(parts[1]))
			y.append(float(parts[2]))
			z.append(float(parts[3]))
		except ValueError:
			print ''

	file.close();
	return (x,y,z)



def difference(a,b):
	# This function assumes the input to be a 3d vector
	return (a[0] - b[0] , a[1] - b[1] , a[2] - b[2] )

def get3Coordinates(x, y, z):
	a = len(x)
	i  = randint(0, a-1)
	j  = randint(0, a-1)
	k  = randint(0, a-1)
	return ( (x[i], y[i], z[i]), (x[j], y[j], z[j]) ,(x[k], y[k], z[k]) ) 


def getOrbitPlaneDirection(a,b,c):

	# there are two possible directions for any plane.
	# say, if one is x, other one should be -x .
	# This function always returns the direction with positive value
	# of z- coordinate, just as a convention
	# (x is a unit vector)

	x = np.cross (np.array( difference(a, b )), np.array( difference(c,b) )) 
	mod = np.linalg.norm(x)

	# returns the unit vector with positive z component
	if  x[2]/mod<0:
		return (-1*x[0]/mod, -1*x[1]/mod, -1*x[2]/mod)
	else : 
		return (x[0]/mod, x[1]/mod, x[2]/mod)

def getInclination(h):
	# h is the direction of orbital plane with respect to ECI frame
	return (math.acos(h[2] /np.linalg.norm(h)) * 180.0 / math.pi)

def getRaan(h):
	# h is the direction of orbital plane with respect to ECI frame
	n = np.cross(np.array([0,0,1]), np.array(h))
	return (math.acos(n[0] /np.linalg.norm(n)) * 180.0 / math.pi)

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
	moda = np.linalg.norm(a)
	modb = np.linalg.norm(b)
	modc = np.linalg.norm(c)
	aXb = np.cross(a,b)
	bXc = np.cross(b,c)
	cXa = np.cross(c,a)

	if abs(np.dot( a, aXb)) > 0.00001:
		return 'vectors a, b, c are not coplanar'
   	
	N = aXb*modc + bXc*moda + cXa*modb
	D = aXb + bXc + cXa
	S = a*(modb - modc) + b*(modc - moda) + c*(moda - modb)

	va = ((mu/(np.linalg.norm(N)* np.linalg.norm(D)))**0.5) * (S + np.cross (D,a)/moda) 
	vb = ((mu/(np.linalg.norm(N)* np.linalg.norm(D)))**0.5) * (S + np.cross (D,b)/modb) 
	vc = ((mu/(np.linalg.norm(N)* np.linalg.norm(D)))**0.5) * (S + np.cross (D,c)/modc) 

	return (va, vb, vc)

def getE(r,v):
	r = np.array(r)
	v = np.array(v)
	modr = np.linalg.norm(r)
	modv = np.linalg.norm(v)
	e = ((modv**2 - (mu/modr))*r  - (np.dot(r,v))*v )/mu
	return e

def getPeriapsisArgument(h,e):
	n = np.cross(np.array([0,0,1]), np.array(h))
	return math.acos(  np.dot(n,e) / np.linalg.norm(n)* np.linalg.norm(e)) * 180 / math.pi

if __name__ == '__main__':

	x, y, z =  readXYZ("trackdata.csv");
	
	plt.plot(0, 0, "o")
	plt.plot(x, y)
	plt.grid()
	#plt.show()


	a, b, c = get3Coordinates(x,y,z) # returns lists a,b,c
	
	planeDir = getOrbitPlaneDirection(a,b,c)

	va, vb, vc = getVelocity(a,b,c) # returns array

	e = getE(a, va) # returns array

	print getPeriapsisArgument(planeDir,e)
	