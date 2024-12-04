# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016

@author: Hossam Faris
"""
import matplotlib.pyplot as plt
import random
import numpy
import math
from solution import solution
import time
import pandas as pd

def objf(x):
    a = numpy.zeros(16)
    d_plus = -5000
    d_minus = -5000
    s1 = ''
    for i in range(64):
        if(x[i]>=0.5):
            s1+= '1'
        else:
            s1+= '0'
    for i in range(0,64,4):
        s = s1[i:i+4]

        a[int(i/4)] = int(s,2)/16
        D_p = ((int(i/4)+1)/16)-a[int(i/4)]
        D_m = a[int(i/4)]-((int(i/4))/16)
        d_plus = max(D_p,d_plus)
        d_minus = max(D_m,d_minus)
    lam1 = max(d_plus,d_minus)
    zero =[]
    ones = []
    last_zero = -1
    last_one = -1
    for i in range(64):
        if(s1[i]=='0'):
            if(last_zero>=0):
                zero.append(i-last_zero-1)
            last_zero = i
        else:
            if(last_one>=0):
                ones.append(i-last_one-1)
            last_one = i
    zero.sort()
    ones.sort()
    l = max(zero[-1],ones[-1])
    mid = l//2
    freq = numpy.zeros(2)
    c_freq = numpy.ones(2)
    for i in range(len(zero)):
        if(zero[i]<=mid):
            freq[0] += 1
        else:
            freq[1] += 1
    for i in range(len(ones)):
        if(ones[i]<=mid):
            freq[0] += 1
        else:
            freq[1] += 1
    c_freq[0] = freq[0]/62
    f_x = numpy.zeros(2)
    f_x[0] = 1 - (0.1**(mid+1))
    f_x[1] = 1- (0.1**(l+1))
    lam2 = max(abs(f_x[0] - c_freq[0]), abs(f_x[1]-c_freq[1]))
    return 1/lam1 + lam2
    

def GWO(lb,ub,dim,SearchAgents_no,Max_iter):
    
    
    #Max_iter=1000
    #lb=-100
    #ub=100
    #dim=30  
    #SearchAgents_no=5
    
    # initialize alpha, beta, and delta_pos
    Alpha_pos=numpy.zeros(dim)
    Alpha_score=float(0)
    
    Beta_pos=numpy.zeros(dim)
    Beta_score=float(0)
    
    Delta_pos=numpy.zeros(dim)
    Delta_score=float(0)

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    #Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = numpy.random.uniform(0,1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    Convergence_curve=numpy.zeros(Max_iter)
    s=solution()

     # Loop counter
    print("GWO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    result_df = pd.DataFrame([], columns=['acc', 'error', 'count', 'time'])
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i,j]=numpy.clip(Positions[i,j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:])                  
            
            # Update Alpha, Beta, and Delta
            if fitness>Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness<Alpha_score and fitness>Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness<Alpha_score and fitness<Beta_score and fitness>Delta_score): 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
            
        
        
        
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
                
            
        
        
        Convergence_curve[l]=Alpha_score;
        #feat=objf(Positions[i,:])[1]
        if (l%1==0):
               #print(['At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)+'  '+str(feat)]);
               result_df.loc[l] = [Alpha_score, 1 - Alpha_score, str(l), time.time()]

    #result_df.to_csv("GWO.csv", mode='a')
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    #s.features=feat
    s.optimizer="GWO"
    s.objfname=objf.__name__
    s.accuracy=Alpha_score
    s.error=1-Alpha_score
    s.bestIndividual = Alpha_pos
    return s
    
    
pos_best_g = GWO(0,1,64,5,1000)
y1 = pos_best_g.convergence
pos_best_g = pos_best_g.bestIndividual
s=''
for i in range(len(pos_best_g)):
    if(pos_best_g[i]>=0.5):
        pos_best_g[i] = 1
        s += '1'
    else:
        pos_best_g[i] = 0
        s += '0'
x1= numpy.arange(1000)
plt.plot(x1,y1)
print(y1)
plt.savefig("GWO.png")
plt.show()
print(s)



    

