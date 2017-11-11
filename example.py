# -*- coding: utf-8 -*-

from AdversarialPredictor import AdversarialPredictionClassifier

#dependencies
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


#download dataset
iris_dataset = datasets.load_iris()
#limit features to sepal and petal lenght
X = iris_dataset["data"][:,[0,2]]
y = iris_dataset["target"]
#limited sample with virginica and versicolor species
Xt = X[y<>0]
yt = y[y<>0]
Xs = X[y==0] #setosa sample

#2 class model fit
m_logr = LogisticRegression()
m_logr.fit(Xt,yt)

#3 class plot
plt.title('3-class plot')
plt.scatter(X[:,0][y==0],X[:,1][y==0], label="setosa",color="green",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==1],X[:,1][y==1], label="versicolor",color="red",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==2],X[:,1][y==2], label="virginica",color="blue",edgecolors=(0, 0, 0))
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.xlim(4,8)
plt.ylim(0.5,7.5)
plt.legend()
plt.savefig("3-class.pdf")

#2 class plot
plt.title('2-class plot')
plt.scatter(X[:,0][y==1],X[:,1][y==1], label="versicolor",color="red",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==2],X[:,1][y==2], label="virginica",color="blue",edgecolors=(0, 0, 0))
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.xlim(4,8)
plt.ylim(0.5,7.5)
plt.legend()
plt.savefig("2-class.pdf")


#2 class logistic regression decision boundary
xx, yy = np.mgrid[4:8:.01, 0.5:7.5:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = m_logr.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.title('2-class Logistic Regression Decision Boundary')
plt.scatter(X[:,0][y==1],X[:,1][y==1], label="versicolor",color="red",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==2],X[:,1][y==2], label="virginica",color="blue",edgecolors=(0, 0, 0))
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.xlim(4,8)
plt.ylim(0.5,7.5)
plt.legend()
plt.contour(xx, yy, probs, levels=[.5], colors="black")
plt.savefig("2-class-logr.pdf")


#3 class logistic regression probabilistic decision boundary
xx, yy = np.mgrid[4:8:0.01, 0.5:7.5:0.01]
Z = m_logr.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
Z = Z.reshape(xx.shape)
image = plt.imshow(Z.T, interpolation='nearest',
                   extent=(4, 8, 0.5, 7.5),
                   aspect='auto', origin='lower', cmap=plt.cm.RdBu)
plt.colorbar(image)
plt.title('2-class Logistic Regression\n Probabilistic Decision Boundary')
plt.scatter(X[:,0][y==1],X[:,1][y==1], label="versicolor",color="red",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==2],X[:,1][y==2], label="virginica",color="blue",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==0],X[:,1][y==0], label="setosa",color="green",edgecolors=(0, 0, 0))
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.xlim(4,8)
plt.ylim(0.5,7.5)
plt.legend()
plt.savefig("2-class-logr-prob.pdf")


#gaussian process decision map
xx, yy = np.mgrid[4:8:0.05, 0.5:7.5:0.05]
kernel = 1.0 * RBF([1.0, 1.0])#rbf_anisotropic
m_gpc = GaussianProcessClassifier(kernel=kernel).fit(Xt, yt)
Z = m_gpc.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
Z = Z.reshape(xx.shape)
image = plt.imshow(Z.T, interpolation='nearest',
                   extent=(4, 8, 0.5, 7.5),
                   aspect='auto', origin='lower', cmap=plt.cm.RdBu)
plt.scatter(X[:,0][y==1],X[:,1][y==1], label="versicolor",color="red",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==2],X[:,1][y==2], label="virginica",color="blue",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==0],X[:,1][y==0], label="setosa",color="green",edgecolors=(0, 0, 0))
plt.colorbar(image)
plt.title("2-class RBF Gaussian Process Classifier\n Decision Map")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.xlim(4,8)
plt.ylim(0.5,7.5)
plt.legend()
plt.savefig("2-class-gpc-prob.pdf")

#adversarial predictor classifier
m_apc = AdversarialPredictionClassifier()
m_apc.fit(Xt,yt)

xx, yy = np.mgrid[4:8:0.05, 0.5:7.5:0.05]
Z = m_apc.predict_conf(np.vstack((xx.ravel(), yy.ravel())).T)
Z_a = []
for i in Z:
    if i[0][0]>i[0][1]:
        Z_a.append(0.5-i[1]/2.)
    else:
        Z_a.append(0.5+i[1]/2.)
               
Z_a = np.array(Z_a)
Z_a = Z_a.reshape(xx.shape)
image = plt.imshow(Z_a.T, interpolation='nearest',
                   extent=(4, 8,0.5, 7.5),
                   aspect='auto', origin='lower', cmap=plt.cm.RdBu)
plt.scatter(X[:,0][y==1],X[:,1][y==1], label="versicolor",color="red",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==2],X[:,1][y==2], label="virginica",color="blue",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==0],X[:,1][y==0], label="setosa",color="green",edgecolors=(0, 0, 0))
plt.colorbar(image)
plt.title("2-class Adversarial Predictor Classifier\n Confidence Map")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.xlim(4,8)
plt.ylim(0.5,7.5)
plt.legend()
plt.savefig("2-class-apc-prob4.pdf")

#visual description of apc
pred = m_apc.predict_conf(Xs[1].reshape(1,2))
X_cand = np.array(m_apc.X_cand)

plt.title('Predicting new sample x')
plt.scatter(X[:,0][y==1],X[:,1][y==1], label="versicolor",color="red",edgecolors=(0, 0, 0))
plt.scatter(X[:,0][y==2],X[:,1][y==2], label="virginica",color="blue",edgecolors=(0, 0, 0))
plt.scatter(Xs[1][0],Xs[1][1], label="new sample to be predicted",color="grey",edgecolors=(0, 0, 0),marker="x")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.xlim(4,8)
plt.ylim(0.5,7.5)
plt.legend()
plt.savefig("new-sample.pdf")


plt.title('Base classifier prediction: versicolor')
plt.scatter(X[:,0][y==1],X[:,1][y==1], label="versicolor",color="red",edgecolors=(0, 0, 0))
plt.scatter(Xs[1][0],Xs[1][1], label="new sample to be predicted",color="grey",edgecolors=(0, 0, 0),marker="x")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.xlim(4,8)
plt.ylim(0.5,7.5)
plt.legend()
plt.savefig("versicolor-predicted.pdf")

plt.title('Adversarial sample is generated')
plt.scatter(X[:,0][y==1],X[:,1][y==1], label="versicolor",color="red",edgecolors=(0, 0, 0))
plt.scatter(X_cand[:,0],X_cand[:,1], label="adversarial sample",color="grey",edgecolors=(0, 0, 0),marker="x")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.xlim(4,8)
plt.ylim(0.5,7.5)
plt.legend()
plt.savefig("adversarial-sample.pdf")


xx, yy = np.mgrid[4:8:.01, 0.5:7.5:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = m_apc.m_adv.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.title('High accuracy means low confidence')
plt.scatter(X[:,0][y==1],X[:,1][y==1], label="versicolor",color="red",edgecolors=(0, 0, 0))
plt.scatter(X_cand[:,0],X_cand[:,1], label="adversarial sample",color="grey",edgecolors=(0, 0, 0),marker="x")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.xlim(4,8)
plt.ylim(0.5,7.5)
plt.contour(xx, yy, probs, levels=[.5], colors="black")
plt.legend()
plt.savefig("confidence-classifier.pdf")

