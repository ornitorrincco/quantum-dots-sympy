#Calcula la ecuacion maestra en su forma lindbladiana para nano-dots de forma lineal en el regimen coulmbiano
#utilizando la expresion:
# [H,R]_ij = (-i/h) * [H,R] + suma(G[d,i]*R[d,d] - G[i,d]*R[i,i])Kronecker(i,j)
#            - (1/2)*(suma(G[n,d]) + suma(G[k,d]))*R[n,k]*( 1 - Kronecker(i,j))

import numpy as np
from IPython.display import display
from sympy import *
from sympy import summation, oo,Symbol, Integer, symbols, log,latex,I #al usar asterisco no se cargan todas las funciones de la libreria
from sympy import init_session    #esto es lo que hace que se imprima en latex

#calcula el conmutador de 2 matrices
def conmutador(a,b): 
    return a*b - b*a
    
#escribe los elementos de matriz del conmutador (parte izquierda de la ecuacion)
def EDMI(M1,M2):
    g = M1.shape[1]
    a1 = conmutador(M1,M2)
    a2 = zeros(1,len(a1))
    l = 0
    for i in range (0,g):
        for k in range (0,g):
             a2[l] = Symbol('[H,\\rho]'+'_{'+str(i)+str(k)+'}') #la H y Rho se cambiana  mano 
             l = l + 1
    return a2
    #escribe los elementos de matriz del conmutado la derivada de rho(parte izquierda de la ecuacion en latex)
def drhosI(M1,M2):
    g = M1.shape[1]
    a1 = conmutador(M1,M2)
    a2 = zeros(1,len(a1))
    l = 0
    for i in range (0,g):
        for k in range (0,g):
             a2[l] = Symbol('\dot{\\rho}'+'_{'+str(i)+str(k)+'}') #la H y Rho se cambiana  mano 
             l = l + 1
    return a2
#calcula los elementos de matriz de derecho del conmutador (parte derecha de la ecuacion)
def EDMD(M1,M2,M3): #M1 es el hamiltoniano, M2 esla matriz de Rhos y M3 es la matriz de Gammas
    a = (-I/h)*conmutador(M1,M2)
    a = a.expand()   
    b = a[:]                 
    return b
#calcula los elementos de matriz de derivada de rho  (parte derecha de la ecuacion, en latex)
def drhosD(M1,M2,M3): #M1 es el hamiltoniano, M2 esla matriz de Rhos y M3 es la matriz de Gammas
    a = (-I/h)*conmutador(M1,M2) +  disipativo(M3,M2)    
    b = a[:]                 
    return b
#imprime los elementos de matriz del conmutador
def printlatexC1(M1,M2,M3): #M1 = H,M2 = R, M3 = G
    RI = (-I/h)*EDMI(M1,M2)
    RD = EDMD(M1,M2,M3) 
    print '\\begin{align*}'
    for n in range (0,len(RI)):
        print latex(RI[n])+'&='+latex(RD[n])+'\\\\' # es necesario el &= para la alinacion en el igual de las ecuaciones
    print '\\end{align*}'    
#imprime los elementos de matriz de l derivada de rho, en formato latex
def printlatexC2(M1,M2,M3): #M1 = H,M2 = R, M3 = G
    RI = drhosI(M1,M2)
    RD = drhosD(M1,M2,M3) 
    print '\\begin{align*}'
    for n in range (0,len(RI)):
        print latex(RI[n])+'&='+latex(RD[n])+'\\\\' # es necesario el &= para la alinacion en el igual de las ecuaciones
    print '\\end{align*}' 
#crea la matriz de rhos para cualquier dimension 
def MRhos(a):
    q = zeros(a)
    for i in range(0,a):
        for k in range(0,a): 
            q[i,k] =Symbol('rho_'+str(i)+str(k))
    return q
    
#crea la matriz de gammas para cualquier dimension
def Gamma(numero):
    if numero > 2 :
        q = zeros(numero)
        p1 = Symbol('Gamma_S')
        p2 = Symbol('Gamma_D')
        q[0,1] = p1
        q[2,0] = p2
        
        return q
    if numero < 3 :
        return "numero invalido, no es posible usar un valor menor que 3"
        

def Hamiltoniano(numero):
        
    if numero > 2:
        H = eye(numero)
        H[0,0] = 0
        for n in range(1,numero):
            H[n,n] = Symbol('epsilon_'+str(n))  
        T = Symbol('T')
        H[1,2] = T
        H[2,1] = T
        return H
        
    if numero < 3:
        return "numero invalido"
        
        
#suma en un intervalo [inicial, final] los elementos de una matriz
def sumaC(c,indice,inicial,final):
    w = 0
    for n in range(inicial,final):
        w = w + c[indice,n]
    return w
#suma la primera parte disipativa de la ecuacion de lindblad
def sumaD1(M1,M2,n):   #M1 es la matriz Gamma y M2 es la matriz de rhos,donde n, es el indice i
    numero = M2.shape[0]   #obtiene las dimensiones de las matrices
    suma = 0
    for k in range(0,numero):
        if k != n :
            suma = suma + M1[k,n]*M2[k,k] - M1[n,k]*M2[n,n]
    return suma
    #suma de la segunda parte de la parte disipativa de la lindblad
def sumaD2(M1,M2,n,k): #M1 debe es gama y M2 rhos, n es el indice i, y k el j
    numero = M2.shape[0]
    suma1 = 0
    suma2 = 0
    suma = 0
    for p in range(0,numero):
        if p != n:
            suma1 = suma1 + M1[n,p]  
        
    for p in range(0,numero):
        if p != k:
            suma2 = suma2 + M1[k,p]
            
    suma = suma1 + suma2
    suma = suma*M2[n,k]
    
#    D[n,k] - (1/2)*(suma(G[n,d]) + suma(G[k,d]))*R[n,k]
    return -suma/2
#obtiene la parte disipativa de la linblad equation
def disipativo(G,R):
    dimension = R.shape[0]
    D = zeros(dimension)
    for n in range(0,dimension):
        for k in range(0,dimension):
            D[n,k] = sumaD1(G,R,n)*KroneckerDelta(n,k)
            D[n,k] =D[n,k] + sumaD2(G,R,n,k)*(1 - KroneckerDelta(n,k))
    return D
#regresa la matriz que al multiplicarse por un vector de Rhos nos da un sistema de ecuaciones
# c = a[0,1].subs('rho_'+str(0)+str(1),1).evalf()
def generador(M1,TM): #luego lo cambio.    M1 es la matriz con los elements, y TM, es R
    dimensiones = len(TM)
    fila =TM.shape[0]
    M1 = M1.expand()
    M1 = M1[:]
    sistema = zeros(dimensiones) 
    for n in range(0,dimensiones):
         terminos = len(M1[n].args)
         for m in range(0,terminos):
            termino = M1[n].args[m]
            
            for k in range(0,dimensiones):
                indice1 = int(floor(k/fila))
                indice2 = k%fila
                c = termino.subs('rho_'+str(indice1)+str(indice2),1).evalf() #esto esta mal implementado, hay una mejor forma
                if termino != c: 
                    sistema[n,k] = sistema[n,k] + c               
    return sistema
#multiplica a la matriz de 9x9 por un vector columna de Rhos
def generador1(M1,TM): #luego lo cambio.    M1 son los elementos de matriz, y TM, es la matriz de Rhos
    a = generador(M1,TM)
    dimensiones = M1.shape[0]
    b = MRhos(dimensiones)[:] 
    numero = len(b)
    for n in range(0,numero):
        a[:,n] = a[:,n]*b[n]
        
    return a            
def determinante(M1):
    return 0
     
init_session()
numeroQD = 2   #numero de  QD

#declaracion de constantes
h = symbols('\hbar')

  # declaracion de matrices
H = Hamiltoniano(numeroQD + 1)
R = MRhos(numeroQD + 1)
G = Gamma(numeroQD + 1)

#generacion de los elementos de matriz en forma de matriz
conservativa = (-I/h)*conmutador(H,R)   
disipativa = disipativo(G,R)
sistema2 = conservativa + disipativa

#crea el sistema de ecuaciones en forma matricial (solo coeficientes)
sistema = generador(sistema2,R)

#crea el sistema de ecuaciones en forma matricial (con rhos)
sistema1 = generador1(sistema2,R)
#display(sistema)
        