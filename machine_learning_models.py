#Imports
import pylab as pb
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize as opt

from sklearn.linear_model import LinearRegression
from math import pi
from scipy.spatial.distance import cdist
from numpy.linalg import inv
from scipy.stats import multivariate_normal


#Prior distribution over W
W = [-1.3,0.5]

mu = -0.4
variance = 1.62

sigma = math.sqrt(variance)

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x,mlab.normpdf(x, mu, sigma))
plt.show()

#sampling from the multivariate distribution
n = np.linspace(-5,5, 200)
x1 = x.reshape(-1,1)

mu1 = np.zeros(x.shape)

#compute covariance matrix
K1 = np.exp(-cdist(x1,x1)**(2/5))
f1 = np.random.multivariate_normal(mu1.flatten(),K1,5)

#plotting samples
plt.plot(x1,f1.T)
plt.show()

#Picking a single data point and then visualising the posterior distribution over W
N= 50

# create multivariate distribution
pdf = multivariate_normal([0,0], [[1 , 0],[0,1]])

# generate points along axis
x = np.linspace(-5,5,N)
y = np.linspace(-5,5,N)

# get all combinations of the points
x1p,x2p = np.meshgrid(x,y)
pos = np.vstack((x1p.flatten(),x2p.flatten()))
pos = pos.T

# evaluate pdf at points
Z = pdf.pdf(pos)
Z = Z.reshape(N,N)

# plot contours
plt.ylabel('w1')
plt.xlabel('w0')
plt.title('Prior over w')
pdf_c = plt.contour(x1p,x2p,Z,15)
plt.show()


phi = np.zeros((201,2))

x =  np.zeros((201,1))
y =  np.zeros((201,1))

phi[:,0].fill(1)

alpha = 2
beta = 1/0.3

for i in range(201):
    x[i] = -1 + i*0.01
    phi[i,1] = x[i]
    y[i] = -1.3*x[i]+0.5+np.random.normal(0, 0.3)

rand = random.sample(range(1, 201), 50)

a = phi
b = rand
c = [ a[i] for i in b]
d = [ y[i] for i in b]

c = np.array(c)
d = np.array(d)


s_n = np.linalg.inv(alpha*np.identity(2) + beta*np.matmul(phi.T,phi))
m_n = np.matmul(s_n, beta*np.matmul(phi.T,y))


s_1 = np.linalg.inv(alpha*np.identity(2) + beta*np.matmul(phi[0:1].T.reshape(2,1),phi[0:1].reshape(1,2)))
m_1 = np.matmul(s_1, beta*np.matmul(phi[0:1].T.reshape(2,1),y[0:1]))


s_2 = np.linalg.inv(alpha*np.identity(2) + beta*np.matmul(phi[0:2].T.reshape(2,2),phi[0:2].reshape(2,2)))
m_2 = np.matmul(s_2, beta*np.matmul(phi[0:2].T.reshape(2,2),y[0:2]))


s_3 = np.linalg.inv(alpha*np.identity(2) + beta*np.matmul(phi[0:3].T.reshape(2,3),phi[0:3].reshape(3,2)))
m_3 = np.matmul(s_3, beta*np.matmul(phi[0:3].T.reshape(2,3),y[0:3]))


s_50 = np.linalg.inv(alpha*np.identity(2) + beta*np.matmul(c.T.reshape(2,50),c.reshape(50,2)))
m_50 = np.matmul(s_50, beta*np.matmul(c.T.reshape(2,50),d))

#Sampled after 1 data point
f11 = np.random.multivariate_normal(m_1.reshape(2,),s_1)
f12 = np.random.multivariate_normal(m_1.reshape(2,),s_1)
f13 = np.random.multivariate_normal(m_1.reshape(2,),s_1)
f14 = np.random.multivariate_normal(m_1.reshape(2,),s_1)
f15 = np.random.multivariate_normal(m_1.reshape(2,),s_1)

w = np.linspace(-1.5,0,100)

plt.scatter(x[0:1],y[0:1])
plt.plot(w,f11[1] * w + f11[0])
plt.plot(w,f12[1] * w + f12[0])
plt.plot(w,f13[1] * w + f13[0])
plt.plot(w,f14[1] * w + f14[0])
plt.plot(w,f15[1] * w + f15[0])

print(x[0])
print(y[0])

plt.title('Samples with 1 data point')
plt.show()

#Sampled after 50 data points
f501 = np.random.multivariate_normal(m_50.reshape(2,),s_50)
f502 = np.random.multivariate_normal(m_50.reshape(2,),s_50)
f503 = np.random.multivariate_normal(m_50.reshape(2,),s_50)
f504 = np.random.multivariate_normal(m_50.reshape(2,),s_50)
f505 = np.random.multivariate_normal(m_50.reshape(2,),s_50)

q = np.linspace(-2,2,100)

plt.ylabel('y')
plt.xlabel('x')
plt.title('Samples with 50 data points')
plt.scatter(c[:,1],d)
plt.plot(q,f501[1]*q+f501[0])
plt.plot(q,f502[1]*q+f502[0])
plt.plot(q,f503[1]*q+f503[0])
plt.plot(q,f504[1]*q+f504[0])
plt.plot(q,f505[1]*q+f505[0])
plt.show()

# create multivariate distribution
pdf = multivariate_normal(m_1.reshape(2,), s_1)

# generate points along axis
x1 = np.linspace(-4,4,N)
y1 = np.linspace(-4,4,N)

# get all combinations of the points
x1p,x2p = np.meshgrid(x1,y1)
pos = np.vstack((x1p.flatten(),x2p.flatten()))
pos = pos.T

# evaluate pdf at points
Z = pdf.pdf(pos)
Z = Z.reshape(N,N)

# plot contours
plt.ylabel('w1')
plt.xlabel('w0')
plt.title('Posterior over 1 data point')
pdf_c = plt.contour(x1p,x2p,Z,15)
plt.show()

# create multivariate distribution
pdf = multivariate_normal(m_50.reshape(2,), s_50)

# generate points along axis
x1 = np.linspace(-1,1,N)
y1 = np.linspace(-3,1,N)

# get all combinations of the points
x1p,x2p = np.meshgrid(x1,y1)
pos = np.vstack((x1p.flatten(),x2p.flatten()))
pos = pos.T

# evaluate pdf at points
Z = pdf.pdf(pos)
Z = Z.reshape(N,N)

# plot contours
plt.ylabel('w1')
plt.xlabel('w0')
plt.title('Posterior over 50 data points')
pdf_c = plt.contour(x1p,x2p,Z,15)
plt.show()

#Create Data
x = np.linspace(-5,5,200).reshape(-1,1)
mu = np.zeros(x.shape)
lengthscale = 1

K = np.exp(-cdist(x,x)/lengthscale**2)
samples_from_prior = np.random.multivariate_normal(mu.flatten(),K,3)

plt.plot(x, samples_from_prior.T)
plt.show()

Xtest = np.linspace(-math.pi, math.pi, 7).reshape(-1,1)
Ytest = np.linspace(-math.pi, math.pi, 7).reshape(-1,1)

for i in range(7):
    Ytest[i] = np.sin(Xtest[i])+np.random.normal(0, 0.5)

lengthscale = 0.7

#defining a squared exponential kernel
lengthscale = 0.7
def kernel(x,y,lengthscale):
    return np.exp(-cdist(x,y)/(lengthscale**2))


K = kernel(Xtest, Xtest, lengthscale)
KStar = kernel(Xtest, x, lengthscale)
KStarStar = kernel(x, x, lengthscale)

mean = np.matmul(np.matmul(KStar.T, np.linalg.inv(K)),Ytest)
covariance = KStarStar - np.matmul(np.matmul(KStar.T,np.linalg.inv(K)), KStar)

FPosterior = np.random.multivariate_normal(mean.flatten(), covariance, 3)

plt.title('3 samples from the posterior')
plt.plot(x, FPosterior.T)
plt.scatter(Xtest, Ytest, c = 'black', s = 100)
plt.show()

#Sampling from the posterior with error
const = .5
KError = K + const*np.identity(K.shape[0])
meanError = np.matmul(np.matmul(KStar.T, np.linalg.inv(KError)),Ytest)
covarianceError = KStarStar - np.matmul(np.matmul(KStar.T,np.linalg.inv(KError)), KStar)

FPosteriorError = np.random.multivariate_normal(meanError.flatten(), covarianceError, 3)

plt.title('3 samples from the posterior with error')
plt.plot(x, FPosteriorError.T)
plt.scatter(Xtest, Ytest, c = 'black', s = 100)
plt.show()

print(mean.shape)
print(covariance.shape)

n = np.zeros(mu.shape)
m = np.zeros(mu.shape)

for i in range(200):
    n[i] = mean[i] + 1*np.sqrt(covariance[i,i])
    m[i] = mean[i] - 1*np.sqrt(covariance[i,i])

plt.title('Predictive mean with predictive variance σ=1')
plt.plot(x,mean)
plt.plot(x, m,'green')
plt.plot(x, n,'green')
plt.gca().fill_between(x.flatten(),n.flatten(),m.flatten(), alpha = 0.2)
plt.plot(x, mean, 'r')
plt.scatter(Xtest, Ytest)
plt.show()

xd = np.linspace(0, 4*pi, 100)

def fnonlin(x):
    #f non linear function
    val = [x*sin(x), x*cos(x)]
    return val

fnl = np.zeros((100,2))
A = np.zeros((10,2))

for i in range(100):
    fnl[i,0] = xd[i]*np.sin(xd[i])
    fnl[i,1] = xd[i]*np.cos(xd[i])

fnl = fnl.T
A = np.random.normal(0,1,(10,2))
B = np.random.normal(0,1,(10,2))
Y = np.matmul(A, fnl)

def f(x, *args):
    # return the value of the objective at x
    x = x.reshape((10,2))
    C = np.matmul(x, x.T)+ np.eye(10)

    val = math.log(np.linalg.det(C)) + np.matrix.trace(np.matmul(Y.T, np.matmul(inv(C),Y)))
    return val

print("Value of objective at x: {}".format(f(A)))

def dfx(x,*args):
    # return the gradient of the objective at x
    x = x.reshape((10,2))
    C = np.matmul(x, x.T)+ np.eye(10)

    val = np.zeros((10,2))

    for i in range(10):
        for j in range(2):

            J_ij = np.zeros((2,10))
            J_ij[j,i] = 1

            J_ji = np.zeros((10,2))
            J_ji[i,j] = 1


            delWWt = np.matmul(x,J_ij) + np.matmul(J_ji,x.T)

            P = np.matmul(inv(C), delWWt)

            YYt = np.matmul(Y, Y.T)

            CdelC = np.matmul(inv(C), np.matmul(delWWt, inv(C)))

            val[i,j] = np.matrix.trace(P) + np.matrix.trace(np.matmul(YYt,-CdelC))

    return val.flatten()

print("Value of gradient at x: {}".format(dfx(A)))    

x_star = opt.fmin_cg(f,B.flatten(),fprime=dfx)
xy = np.matmul(np.linalg.pinv(x_star.reshape((10,2))),Y)

fig = plt.figure(figsize=(13,20))
ax1 = plt.subplot(211)
ax1 = plt.scatter(xy[0], xy[1])
ax2 = plt.subplot(212, projection='polar')
ax2.scatter(xy[0],xy[1])
plt.show()
