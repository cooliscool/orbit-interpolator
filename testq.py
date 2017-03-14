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

def get3PointCombination():
	return

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


def getPlaneDirection(a,b,c):
	
	
	return
if __name__ == '__main__':

	x, y, z =  readXYZ("trackdata.csv");
	a =  fitEllipse(x,y)


	plt.plot(0, 0, "o")
	plt.plot(x, y)
	plt.grid()
	plt.show()


	print len(x)
	