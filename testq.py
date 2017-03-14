# Algorithm 

# 1) parse x, y, z coordinates from file.
# 2) fit the curve for obtained x, y coordinate set (an ellipse in XY plane)
# 3) choose random 3 (x,y,z) points from the dataset and find direction of orbital plane 
#    Equation of a plane can be found using any three points on the plane. 
# 4) do the same for several combinations of 3 point set (for getting an average estimate)
# 5) rotate the plane to the target plane coordinate system, which is the final orbit 
# 6) calculate the keplerian orbital elements (i,raan, argument_of_periapsis, a, e)

import matplotlib.pylab as plt
import numpy as np
from numpy.linalg import eig, inv
from random import randint

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

def fitEllipse(x,y):
	x = np.array(x)[:,np.newaxis]
	y = np.array(y)[:,np.newaxis]

	D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x))); 
	S = np.dot(D.T,D); 	
	C = np.zeros([6,6])
	C[0,2] = 2; C[1,1] = -1;  C[2,0] = 2;
	E, V =  eig(np.dot(inv(S), C))
	n = np.argmax(np.abs(E))
	a = V[:,n]
	return a

def getPlaneDirection():
	return

def get3Coordinates(x, y, z):
	a = len(x)
	i  = randint(0, a-1)
	j  = randint(0, a-1)
	k  = randint(0, a-1)
	return ( (x[i], y[i], z[i]), (x[j], y[j], z[j]) ,(x[k], y[k], z[k]) ) 

def rotate():
	return

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])


def displayEllipse(a):
	return

def difference(a,b):
	# This function assumes the input to be a 3d vector
	return (a[0] - b[0] , a[1] - b[1] , a[2] - b[2] )


def getPlaneDirection(a,b,c):

	# there are two possible directions for any plane.
	# say, if one is x, other one should be -x .
	# This function always returns the direction with positive value
	# of z- coordinate, just as a convention
	# (x is a unit vector)

	x = np.cross (np.array( difference(a, b )), np.array( difference(c,b) )) 
	mod = np.dot(x,x) ** 0.5
		
	# returns the unit vector with positive z component
	if  x[2]/mod<0:
		return (-1*x[0]/mod, -1*x[1]/mod, -1*x[2]/mod)
	else : 
		return (x[0]/mod, x[1]/mod, x[2]/mod)

	
if __name__ == '__main__':

	x, y, z =  readXYZ("trackdata.csv");
	a =  fitEllipse(x,y)


	plt.plot(0, 0, "o")
	plt.plot(x, y)
	plt.grid()
	#plt.show()


	a, b, c = get3Coordinates(x,y,z)
	
	planeDir = getPlaneDirection(a,b,c)

	print planeDir