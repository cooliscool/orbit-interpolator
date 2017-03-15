# method 2 : fitting an ellipse equation for the coordinates of satellite obtained.

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


def rotate():
	return


	
if __name__ == '__main__':

	x, y, z =  readXYZ("trackdata.csv");
