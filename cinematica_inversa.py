from mpl_toolkits.mplot3d import axes3d
import numpy as np
from scipy.optimize import fsolve
from numpy import newaxis
import matplotlib.pyplot as plt
#from math import *
#from numpy import *
from numpy.linalg import inv

A0 = np.array([[-64.5,64.5,103.0,43.0,-43.0,-103.0],
            [-86.6,-86.6,-5.1,98.7,98.7,-5.19],
            [0.0,0.0,0.0,0.0,0.0,0.0]])

V0 = np.array([[-57.5, 57.5, 92.5, 35.0, -35.0, -92.5],
            [-73.6, -73.6, -12.9, 86.6, 86.6, -12.9],
             [135.0, 135.0, 135.0, 135.0, 135.0, 135.0]])

O = np.array([[-44.0,44.5,93.0,53.0,-53.0,-93.0],
            [-86.6,-86.6,12.1,81.4,81.4,12.1],
            [ 0.0,0.0,0.0,0.0,0.0,0.0]])

U = np.array([[0.0,0.0,0.86,0.86,-0.86,-0.86],
            [-1.0,-1.0,0.5,0.5,0.5,0.5],
            [0.0,0.0,0.0,0.0,0.0,0.0]])
W = np.array([[0,0,1]])

Base_Superior = V0[:,list(range(6))+[0]]
Base_Inferior = O[:,list(range(6))+[0]]

def cos_phi (AA):
  vM = AA - O
  res = np.zeros(6)
  for i in range(6):
    res[i] = (U[:,i].dot(vM[:,i])) / (np.linalg.norm(U[:,i])*np.linalg.norm(vM[:,i]))
  return res

CosPhi0 = cos_phi(A0)

def create_figure():  
  fig = plt.figure()
  ax=fig.add_subplot(111,projection='3d')
  return fig, ax
  
  

def plot_plataforma(ax, AA, VV):
  Base_Superior = VV[:,list(range(6))+[0]]
  ax.plot_wireframe(Base_Superior[0,:],Base_Superior[1,:],Base_Superior[2,:])
  ax.plot(Base_Inferior[0,:],Base_Inferior[1,:],Base_Inferior[2,:])
  for i in range(6):
    D = np.vstack([O[:,i],AA[:,i],VV[:,i]])
    ax.plot(D[:,0],D[:,1],D[:,2],'.-')
  



def calc_RT(alpha, beta, gamma, px, py, pz):

  alpha = alpha * np.pi / 180.0
  beta  = beta  * np.pi / 180.0
  gamma = gamma * np.pi / 180.0

  RX = np.matrix([[1,  0             , 0            ,    0],
               [0,  np.cos(alpha) ,-np.sin(alpha),    0],
               [0,  np.sin(alpha) , np.cos(alpha),    0],
               [0,  0             , 0            ,    1]])

  RY = np.matrix([[np.cos(beta),  0,  -np.sin(beta) ,   0],
               [0           ,  1,    0           ,   0],
               [np.sin(beta),  0,    np.cos(beta),   0],
               [0           ,  0,    0           ,   1]])

  RZ = np.matrix([[np.cos(gamma), -np.sin(gamma),  0,  0],
               [np.sin(gamma),  np.cos(gamma),  0,  0],
               [0            ,  0            ,  1,  0],
               [0            ,  0            ,  0,  1]])

  Tras = np.matrix([[1,0,0,px],
                  [0,1,0,py],
                  [0,0,1,pz],
                 [0,0,0,1]])

  return Tras*RZ*RY*RX
  
def calc_V(RT):
  vertices_ampliado = RT.dot(np.vstack([V0,np.ones((1,6))]))
  return np.array(vertices_ampliado[0:3,:])

def norma_columnas_cuadrado (B):
  C = np.array(B)
  return (sum([fila for fila in C*C]))

def calcTheta(AA):
  M = AA - O
  theta = np.zeros(6)
  for i in range(6):
    theta[i] = np.arccos(M[:,i].dot(W[0])/np.linalg.norm(M[:,i]))
  return theta


def Finverso(AA, V):
  A = AA.reshape(3,6)

  
  L_B2 = norma_columnas_cuadrado(V0 - A0)
  L_M2 = norma_columnas_cuadrado(A0  - O)
  L_U2 = norma_columnas_cuadrado(U)
  
#  Delta_r = A - O
#  cos_theta = W.dot(Delta_r) / (np.linalg.norm(W) * norma_columnas(Delta_r))
#  theta = np.arccos(cos_theta)
  
  
  
  # calculamos las ecuaciones... Consideramos A como una matriz de (3,6). 
  eqnsLargoBarras = norma_columnas_cuadrado(V - A) - L_B2
  eqnsLargoManivelas = norma_columnas_cuadrado(A - O) - L_M2
  eqnsPhiConstante = cos_phi(A) - CosPhi0
  
  return np.hstack((eqnsLargoBarras,eqnsLargoManivelas,eqnsPhiConstante))
    #eqnsCosTheta = 
     
        # np.linalg.norm(W) =  1.0
   #     F[3][i] = (A[:,i]-O[:,i]).dot(W[:,0]) - np.sqrt(L_M[i] * 1.0) * np.cos(theta[0][i]) 
        
  
  
 #       F_prima[:,:]=np.array([[2*(A[0][i]-vertices[0][i]),2*(A[1][i]-vertices[1][i]),2*(A[2][i]-vertices[2][i]),0],
 #                               [2*(A[0][i]-O[0][i]),2*(A[1][i]-O[1][i]),2*(A[2][i]-O[2][i]),0],
 #                               [U[0][i],U[1][i],U[2][i],0],
 #                               [W[0][0],W[1][0],W[2][0],L_M[0][i]*np.sin(theta[0][i])]]) 
 #       
 #     J.append(F_prima)
 #     J_1=inv(J[i])
 #     Fprima_1.append(J_1)
 #     X_n[0:3,i]=A[:,i]     
 #     X_n[3,i]=theta[0][i]  
 #     X_n_1[:,i]=X_n[:,i]-(Fprima_1[i].dot(F[:,i]))
 #     A[:,i]=X_n_1[0:3,i]
 #    theta[0][i]=X_n_1[3,i] 
      
      
      
      
  
  
  
  
  
  
