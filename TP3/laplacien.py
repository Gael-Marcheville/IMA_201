#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22 May 2019

@author: M Roux
"""

############## le clear('all') de Matlab
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
################################"

import math
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

from scipy import ndimage
from scipy import signal

from skimage import io

##############################################

# -*- coding: utf-8 -*-
#
#
#
#########################################################################
u"""
lecture, écriture et affichage d'images à forte dynamique, réeles ou complexes (radar)
"""

MRLABVERSION=u"V0.1  Version du 21 mai 2019"

#############################################################################
import numpy as np

import math

globalparamnotebook=0

##############################################################################
#
def version():
    return MRLABVERSION
##############################################################################
#############################
def notebook(*therest):
    u'''
    Sans argument : modifie certains affichages pour les notebooks

    Avec argument :

        ==0 : affichage normal

        ==1 : affichage pour notebook
    '''
    global globalparamnotebook
    globalparamnotebook=1
    if len(therest)==1 :
        globalparamnotebook=therest[0]

#General IO functions

################################################
# 22 mai 2019
def dericheGradX(ima,alpha):


    nl,nc=ima.shape
    ae=math.exp(-alpha)
    c=-(1-ae)*(1-ae)/ae

    b1=np.zeros(nc)
    b2=np.zeros(nc)

    gradx=np.zeros((nl,nc))


#gradx=np.zeros(nl,nc)
    for i in range(nl):

        l=ima[i,:].copy()
        b1[0]=l[2]
        b1[1]=l[2]
        b2[nc-2]=l[nc-3]
        b2[nc-1]=l[nc-3]

        for j in range(2,nc):
            b1[j] = l[j-1] + 2*ae*b1[j-1] - 2*ae*ae*b1[j-2]

        for j in range(nc-3,-1,-1):
            b2[j] = l[j+1] + 2*ae*b2[j+1] - 2*ae*ae*b2[j+2]

        gradx[i,:]=c*ae*(b1-b2);

    return gradx


################################################
# 22 mai 2019
def dericheGradY(ima,alpha):


    nl,nc=ima.shape
    ae=math.exp(-alpha)
    c=-(1-ae)*(1-ae)/ae

    b1=np.zeros(nl)
    b2=np.zeros(nl)

    grady=np.zeros((nl,nc))

    for i in range(nc):
        l=ima[:,i].copy()

        b1[0]=l[2]
        b1[1]=l[2]
        b2[nc-2]=l[nc-3]
        b2[nc-1]=l[nc-3]

        for j in range(2,nc):
            b1[j] = l[j-1] + 2*ae*b1[j-1] - 2*ae*ae*b1[j-2]

        for j in range(nc-3,-1,-1):
            b2[j] = l[j+1] + 2*ae*b2[j+1] - 2*ae*ae*b2[j+2]

        grady[:,i]=c*ae*(b1-b2);


    return grady


################################################
# 22 mai 2019
def sobelGradX(ima):

    nl,nc=ima.shape
    gradx=np.zeros((nl,nc))


#gradx=np.zeros(nl,nc)
    for i in range(1,nl-1):
        for j in range(1,nc-1):
            gradx[i,j]=ima[i-1,j+1]+2*ima[i,j+1]+ima[i+1,j+1]-ima[i-1,j-1]-2*ima[i,j-1]-ima[i+1,j-1];

    return gradx

################################################
# 22 mai 2019
def sobelGradY(ima):

    nl,nc=ima.shape
    grady=np.zeros((nl,nc))


#gradx=np.zeros(nl,nc)
    for i in range(1,nl-1):
        for j in range(1,nc-1):
            grady[i,j]=ima[i+1,j-1]+2*ima[i+1,j]+ima[i+1,j+1]-ima[i-1,j-1]-2*ima[i-1,j]-ima[i-1,j+1];

    return grady
################################################
# 27 juilet 2019
def toto(a):
    print ('a : %f'%a)

################################################
# 22 mai 2019

def normeGradient(gradx,grady):

    nl,nc=gradx.shape
    norme=np.zeros((nl,nc))

    for i in range(nl):
        for j in range(nc):
            norme[i,j]=np.sqrt(gradx[i,j]*gradx[i,j]+grady[i,j]*grady[i,j])

    return norme

################################################
# 22 mai 2019
def dericheSmoothX(ima,alpha):


    nl,nc=ima.shape
    ae=math.exp(-alpha)
    c=(1-ae)*(1-ae)/(1+2*alpha*ae-ae*ae)


    b1=np.zeros(nc)
    b2=np.zeros(nc)

    smoothx=np.zeros((nl,nc))


#gradx=np.zeros(nl,nc)
    for i in range(nl):
        l=ima[i,:].copy()
        for j in range(2,nc):
            b1[j]=c*(l[j]+ae*(alpha-1)*l[j-1])+2*ae*b1[j-1]-ae*ae*b1[j-2]
        b1[0]=b1[2]
        b1[1]=b1[2]
        for j in range(nc-3,-1,-1):
            b2[j]=c*(ae*(alpha+1)*l[j+1]-ae*ae*l[j+2])+2*ae*b2[j+1]-ae*ae*b2[j+2]
        b2[nc-1]=b2[nc-3]
        b2[nc-2]=b2[nc-3]
        smoothx[i,:]=b1+b2;


    return smoothx

################################################
# 22 mai 2019
def dericheSmoothY(ima,alpha):


    nl,nc=ima.shape
    ae=math.exp(-alpha)
    c=(1-ae)*(1-ae)/(1+2*alpha*ae-ae*ae)

    b1=np.zeros(nl)
    b2=np.zeros(nl)

    smoothy=np.zeros((nl,nc))

    for i in range(nc):
        l=ima[:,i].copy()
        for j in range(2,nl):
            b1[j]=c*(l[j]+ae*(alpha-1)*l[j-1])+2*ae*b1[j-1]-ae*ae*b1[j-2]
        b1[0]=b1[2]
        b1[1]=b1[2]
        for j in range(nl-3,-1,-1):
            b2[j]=c*(ae*(alpha+1)*l[j+1]-ae*ae*l[j+2])+2*ae*b2[j+1]-ae*ae*b2[j+2]
        b2[nl-1]=b2[nl-3]
        b2[nl-2]=b2[nl-3]

        smoothy[:,i]=b1+b2;


    return smoothy


################################################
# 22 mai 2019
def maximaDirectionGradient(gradx,grady):


    nl,nc=gradx.shape

    norme=np.sqrt(gradx*gradx+grady*grady)+0.1

    gradx=np.divide(gradx,norme)
    grady=np.divide(grady,norme)

    contours=np.zeros((nl,nc),dtype=int);

    for i in range(1,nl-1):
        for j in range(1,nc-1):
            G1=interpolationbilineaire(norme,i+grady[i,j],j+gradx[i,j]);
            G2=interpolationbilineaire(norme,i-grady[i,j],j-gradx[i,j]);
            if norme[i,j]>=G1 and norme[i,j]>=G2:
                contours[i,j]=1
            else:
                contours[i,j]=0

    return contours

################################################
# 22 mai 2019
def interpolationbilineaire(ima,l,c):

    l,c

    l1=l-np.floor(l)
    l2=np.ceil(l)-l
    c1=c-np.floor(c)
    c2=np.ceil(c)-c


    ll=np.uint32(np.floor(l))
    cc=np.uint32(np.floor(c))

    val=ima[ll,cc]*l2*c2+ima[ll+1,cc]*l1*c2+ima[ll,cc+1]*l2*c1+ima[ll+1,cc+1]*l1*c1

    return val



############################################################################
# Septembre 2017 : une liste de paramètres sont requis
#

##############################################"

############## le close('all') de Matlab
plt.close('all')
################################"


ima=io.imread('C:/Users/march/Desktop/Tetech_cours/2A/IMA_201/TP/TP_30092022/cell.tif')
alpha=0.5

gradx=dericheGradX(dericheSmoothY(ima,alpha),alpha)
grady=dericheGradY(dericheSmoothX(ima,alpha),alpha)

gradx2=dericheGradX(dericheSmoothY(gradx,alpha),alpha)
grady2=dericheGradY(dericheSmoothX(grady,alpha),alpha)



plt.figure('Image originale')
plt.imshow(ima, cmap='gray')


lpima=gradx2+grady2

plt.figure('Laplacien')
plt.imshow(lpima, cmap='gray')


posneg=(lpima>=0)

plt.figure('Laplacien binarisé -/+')
plt.imshow(255*posneg, cmap='gray')

nl,nc=ima.shape
contours=np.uint8(np.zeros((nl,nc)))


for i in range(1,nl):
    for j in range(1,nc):
        if (((i>0) and (posneg[i-1,j] != posneg[i,j])) or
            ((j>0) and (posneg[i,j-1] != posneg[i,j]))):
            contours[i,j]=255


plt.figure('Contours')
plt.imshow(contours, cmap='gray')

#io.imsave('contours.tif',np.uint8(255*valcontours))



