import numpy as np
from Crypto.Cipher import AES
from Crypto import Random
import cv2
import base64
import binascii

def pad(s):
    return s + b"\0" * (AES.block_size - len(s) % AES.block_size)

def gap_test(var):
    maximum=0
    x=-1
    for i in range(64):
        if(var[i]==x):
            temp+=1
        else:
            temp=1
            x=var[i]
        if(temp>=maximum):
            maximum=temp
    return maximum
        
def cross_over(ind1,ind2,bit_size):  
    x = np.random.randint(bit_size)
    a = np.zeros(bit_size)
    a[0:bit_size] = ind1[0:bit_size]
    a[bit_size:]= ind2[bit_size:]
    return a 

def mutate(ind):
    no_of_mutations = np.random.randint(10)
    for i in range(no_of_mutations):
        index = np.random.randint(64)
        ind[index] = (ind[index] + 1)%2
    return ind
       
         
def gakey():
    population = []
    pop = 100
    bit_size = 64
    fitness = []
    curr=0
    generations=100
    mutation_rate=2   
    for i in range(pop):		
        individual = np.random.randint(2,size=bit_size)
        population.append(individual)
        lam1 = gap_test(population[i])
        fitness.append([i])
        fitness[i].append(1/lam1)
    fitness.sort(key=lambda x:x[1])

    while(curr<=generations):
        l = np.array(fitness)
        temp = sum(l,0)
        temp = temp[1]
        prob = []
        new_population=[]
        new_fitness=[]
        prob_chart = -1*np.ones(100)
        #print(prob_chart)
        x1=0
        for i in range(pop):
            x = fitness[i][1]/temp
            x = int(100*x)
            #print(x)
            prob_chart[x1:x1+x]= fitness[i][0]
            x1 = x1+x
        #prob.append([fitness[i][0]])
        #prob[i].append(fitness[i][1]/temp)
        if(x1<99):
            prob_chart[x1:]=prob_chart[x1-1]
        for i in range(pop):
            ind1 = population[int(prob_chart[np.random.randint(100)])]
            ind2 = population[int(prob_chart[np.random.randint(100)])]
            new_child = cross_over(ind1,ind2,bit_size)
            if(np.random.randint(10)>=5):
                new_child = mutate(new_child)
            new_population.append(new_child)
            new_fitness.append([i])
            new_fitness[i].append(1/gap_test(new_child))
        fitness = new_fitness
        population = new_population
        fitness.sort(key=lambda x:x[1])
        
            
        #print(fitness[pop-1][1])
        curr+=1
    #print(population[fitness[pop-1][0]])

    key=population[fitness[pop-1][0]].astype(int)
    s=''
    for i in range(bit_size):
        s += str(key[i])
    print(s)
    return s

 
















