# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:06:34 2016

@author: hossam
"""
import random
import numpy
import matplotlib.pyplot as plt
import time
import math
import sklearn
from numpy import asarray
from sklearn.preprocessing import normalize
from solution import solution
import pandas as pd



def normr(Mat):
   """normalize the columns of the matrix
   B= normr(A) normalizes the row
   the dtype of A is float"""
   Mat=Mat.reshape(1, -1)
     # Enforce dtype float
   if Mat.dtype!='float':
      Mat = asarray(Mat,dtype=float)

   # if statement to enforce dtype float
   B = normalize(Mat,norm='l2',axis=1)
   B=numpy.reshape(B,-1)
   return B

def randk(t):
    if (t%2)==0:
        s=0.25
    else:
         s=0.75
    return s
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

def RouletteWheelSelection(weights):
  accumulation = numpy.cumsum(weights)
  p = random.random() * accumulation[-1]
  chosen_index = -1;
  for index in range (0, len(accumulation)):
    if (accumulation[index] > p):
      chosen_index = index;
      break;
  
  choice = chosen_index;

  return choice



def MVO(lb,ub,dim,N,Max_time):

    "parameters"
    #dim=30
    #lb=-100
    #ub=100
    WEP_Max=1;
    WEP_Min=0.2
    #Max_time=1000
    #N=50
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    
    
    
    Universes = numpy.zeros((N, dim))
    for i in range(dim):
        Universes[:, i] = numpy.random.uniform(0,1, N) * (ub[i] - lb[i]) + lb[i]

    Sorted_universes=numpy.copy(Universes)
    
    convergence=numpy.zeros(Max_time)
     
    
    Best_universe=[0]*dim;
    Best_universe_Inflation_rate= float(0)
    
    
    
    
    s=solution()

    
    Time=1;
    ############################################
    print("MVO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    result_df=pd.DataFrame([],columns=['acc','error','count','time'])
    while (Time<Max_time+1):
    
        "Eq. (3.3) in the paper"
        WEP=WEP_Min+Time*((WEP_Max-WEP_Min)/Max_time)
       
        TDR=1-(math.pow(Time,1/6)/math.pow(Max_time,1/6))
      
        Inflation_rates=[0]*len(Universes)
        
       
       
        for i in range(0,N):
            for j in range(dim):
                Universes[i,j]=numpy.clip(Universes[i,j], lb[j], ub[j])
    
            
    
            Inflation_rates[i]=objf(Universes[i,:])
           
       
               
            if Inflation_rates[i]>Best_universe_Inflation_rate :
                        
                Best_universe_Inflation_rate=Inflation_rates[i]
                Best_universe=numpy.array(Universes[i,:])
             
        
        sorted_Inflation_rates = numpy.sort(Inflation_rates)
        sorted_indexes = numpy.argsort(Inflation_rates)
        
        for newindex in range(0,N):
            Sorted_universes[newindex,:]=numpy.array(Universes[sorted_indexes[newindex],:])   
            
        normalized_sorted_Inflation_rates=numpy.copy(normr(sorted_Inflation_rates))
    
        
        Universes[0,:]= numpy.array(Sorted_universes[0,:])
    
        for i in range(1,N):
            Back_hole_index=i
            for j in range(0,dim):
                r1=random.random()
                
                if r1<normalized_sorted_Inflation_rates[i]:
                    White_hole_index=RouletteWheelSelection(-sorted_Inflation_rates);
    
                    if White_hole_index==-1:
                        White_hole_index=0;
                    White_hole_index=0;
                    Universes[Back_hole_index,j]=Sorted_universes[White_hole_index,j];
            
                r2=random.random() 
                
                
                if r2<WEP:
                    r3=random.random() 
                    if r3<0.5:                    
                        Universes[i,j]=Best_universe[j]+TDR*((ub[j]-lb[j])*random.random()+lb[j]) #random.uniform(0,1)+lb);
                    if r3>0.5:          
                        Universes[i,j]=Best_universe[j]-TDR*((ub[j]-lb[j])*random.random()+lb[j]) #random.uniform(0,1)+lb);
                              
        
        
        
        
        convergence[Time-1]=Best_universe_Inflation_rate
        #feat=objf(Universes[i,:])[1]
        if (Time%1==0):
               #print(['At iteration '+ str(Time)+ ' the best fitness is '+ str(Best_universe_Inflation_rate)+'  '+str(feat)]);
               result_df.loc[Time]=[Best_universe_Inflation_rate,1-Best_universe_Inflation_rate,str(Time),time.time()]
        #result_df.to_csv("MVO.csv",mode='a')

        
        
        
        Time=Time+1
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence
    #s.features=feat
    s.optimizer="MVO"
    s.objfname=objf.__name__
    s.accuracy=Best_universe_Inflation_rate
    s.error=1-Best_universe_Inflation_rate
    s.bestIndividual = Best_universe

    return s

pos_best_g = MVO(0,1,64,5,1000)
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
plt.savefig("MVO.png")
plt.show()
print(s)



