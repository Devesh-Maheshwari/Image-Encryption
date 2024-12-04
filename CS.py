# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:13:28 2016

@author: Hossam Faris
"""
import math
import numpy
import random
import time
from solution import solution
import pandas as pd
import matplotlib.pyplot as plt

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
    


    
def get_cuckoos(nest,best,lb,ub,n,dim):
    
    # perform Levy flights
    tempnest=numpy.zeros((n,dim))
    tempnest=numpy.array(nest)
    beta=3/2;
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta);

    s=numpy.zeros(dim)
    for j in range (0,n):
        s=nest[j,:]
        u=numpy.random.randn(len(s))*sigma
        v=numpy.random.randn(len(s))
        step=u/abs(v)**(1/beta)
 
        stepsize=0.01*(step*(s-best))

        s=s+stepsize*numpy.random.randn(len(s))
    
        for k in range(dim):
            tempnest[j,k]=numpy.clip(s[k], lb[k], ub[k])

    return tempnest

def get_best_nest(nest,newnest,fitness,n,dim,objf):
# Evaluating all new solutions
    tempnest=numpy.zeros((n,dim))
    tempnest=numpy.copy(nest)

    for j in range(0,n):
    #for j=1:size(nest,1),
        fnew=objf(newnest[j,:])
        if fnew>=fitness[j]:
           fitness[j]=fnew
           tempnest[j,:]=newnest[j,:]
        
    # Find the current best

    fmax = max(fitness)
    K=numpy.argmin(fitness)
    bestlocal=tempnest[K,:]
    #feat=objf(newnest[j,:])[1]

    return fmax,bestlocal,tempnest,fitness

# Replace some nests by constructing new solutions/nests
def empty_nests(nest,pa,n,dim):

    # Discovered or not 
    tempnest=numpy.zeros((n,dim))

    K=numpy.random.uniform(0,1,(n,dim))>pa
    
    
    stepsize=random.random()*(nest[numpy.random.permutation(n),:]-nest[numpy.random.permutation(n),:])

    
    tempnest=nest+stepsize*K
 
    return tempnest
##########################################################################


def CS(lb,ub,dim,n,N_IterTotal):

    #lb=-1
    #ub=1
    #n=50
    #N_IterTotal=1000
    #dim=30
    
    # Discovery rate of alien eggs/solutions
    pa=0.25
    
    
    nd=dim
    
    
#    Lb=[lb]*nd
#    Ub=[ub]*nd
    convergence=[]
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # RInitialize nests randomely
    nest = numpy.zeros((n, dim))
    for i in range(dim):
        nest[:, i] = numpy.random.uniform(0,1, n) * (ub[i] - lb[i]) + lb[i]
       
    
    new_nest=numpy.zeros((n,dim))
    new_nest=numpy.copy(nest)
    
    bestnest=[0]*dim;
     
    fitness=numpy.zeros(n) 
    fitness.fill(float(0))
    

    s=solution()

     
    print("CS is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    fmax,bestnest,nest,fitness =get_best_nest(nest,new_nest,fitness,n,dim,objf)
    convergence = [];
    # Main loop counter
    result_df = pd.DataFrame([], columns=['acc', 'error', 'count', 'time'])
    for iter in range (0,N_IterTotal):
        # Generate new solutions (but keep the current best)
     
         new_nest=get_cuckoos(nest,bestnest,lb,ub,n,dim)
         
         
         # Evaluate new solutions and find best
         fnew,best,nest,fitness=get_best_nest(nest,new_nest,fitness,n,dim,objf)
         
        
         new_nest=empty_nests(new_nest,pa,n,dim) ;
         
        
        # Evaluate new solutions and find best
         fnew,best,nest,fitness=get_best_nest(nest,new_nest,fitness,n,dim,objf)
    
         if fnew>fmax:
            fmax=fnew
            bestnest=best
    
         
         if (iter%1==0):
            #print(['At iteration '+ str(iter)+ ' the best fitness is '+ str(fmax)+'  '+str(feat)]);
            result_df.loc[iter] = [fmax, 1 - fmax, str(iter),time.time()]
         convergence.append(fmax)
    result_df.to_csv("CS.csv", mode='a')

    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence
    #s.features=feat
    s.optimizer="CS"
    s.objfname=objf.__name__
    s.accuracy=fmax
    s.error=1-fmax
    s.bestIndividual = bestnest
    return s

pos_best_g = CS(0,1,64,50,1000)
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
plt.savefig("CS.png")
plt.show()
print(s)
    
     
    
    
    




 



