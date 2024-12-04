import random
import numpy as np
import csv
import time
from itertools import repeat

def func1(x):
	a = np.zeros(16)
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
	freq = np.zeros(2)
	c_freq = np.ones(2)
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
	f_x = np.zeros(2)
	f_x[0] = 1 - (0.1**(mid+1))
	f_x[1] = 1- (0.1**(l+1))
	lam2 = max(abs(f_x[0] - c_freq[0]), abs(f_x[1]-c_freq[1]))
	return lam1 + lam2
def dist(a,b):
	val=0
	for i in range(len(a)):
		val=val+(a[i]-b[i])**2
	return val

class Particle:
	def __init__(self,x0):
		self.position_i=[]
		self.velocity_i=[]
		self.pos_best_i=[]
		self.err_best_i=-1
		self.err_i=-1

		for i in range(0,num_dimensions):
			self.velocity_i.append(random.uniform(-1,1))
			self.position_i.append(x0[i])

	def evaluate(self,costFunc):
		self.err_i=costFunc(self.position_i)

		if self.err_i < self.err_best_i or self.err_best_i==-1:
			self.pos_best_i=self.position_i
			self.err_best_i=self.err_i

	def update_velocity(self,pos_best_g):
		w=0.5
		c1=1
		c2=2

		for i in range(0,num_dimensions):
			r1=random.random()
			r2=random.random()

			vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
			vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
			self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

	def update_position(self,bounds):
		for i in range(0,num_dimensions):
			self.position_i[i]=self.position_i[i]+self.velocity_i[i]

			if self.position_i[i]>bounds[1][i]:
				self.position_i[i]=bounds[1][i]

			if self.position_i[i] < bounds[0][i]:
				self.position_i[i]=bounds[0][i]
				
class PSO():
	def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
		global num_dimensions

		probab= 0
		radius= 0.1
		num_dimensions=len(x0)
		global err_best_g
		err_best_g=-1
		global pos_best_g
		pos_best_g=[]
		y1=np.zeros(maxiter)

		swarm=[]
		for i in range(0,num_particles):
			swarm.append(Particle(x0))

		i=0
		while i < maxiter:
			for j in range(0,num_particles):
				swarm[j].evaluate(costFunc)

				if swarm[j].err_i < err_best_g or err_best_g == -1:
					pos_best_g=list(swarm[j].position_i)
					err_best_g=float(swarm[j].err_i)

			for j in range(0,num_particles):
				pos=list(swarm[j].position_i)
				err=float(swarm[j].err_i)
				for k in range(0,num_particles):
					if dist(list(swarm[j].position_i),list(swarm[k].position_i))<=radius:
						if swarm[k].err_i<err:
							err=float(swarm[k].err_i)
							pos=list(swarm[k].position_i)
					else:
						if random.random()<=probab and swarm[k].err_i<err:
							err=float(swarm[k].err_i)
							pos=list(swarm[k].position_i)
				swarm[j].update_velocity(pos)
				swarm[j].update_position(bounds)
			y1[i]=err_best_g
			i+=1
			probab=probab+1.0/maxiter
			


		print('FINAL:')
		with open(r"plot.csv",'a') as f:
			writer=csv.writer(f)
			writer.writerow(y1)
	def key():
		
		return pos_best_g,err_best_g

		#print(err_best_g)
		#print pos_best_g
		#print err_best_g

def psodwcnkey():
	bit_size = 64
	initial=[random.uniform(0,1) for i in range(bit_size)]
	bounds=[tuple(repeat(0,bit_size)),tuple(repeat(1,bit_size))]
	print(bounds)
	start=time.time()
	PSO(func1,initial,bounds,num_particles=20,maxiter=100)
	pos_best_g,error = PSO.key()
	s=''
	for i in range(len(pos_best_g)):
		if(pos_best_g[i]>=0.5):
			pos_best_g[i] = 1
			s += '1'
		else:
			pos_best_g[i] = 0
			s += '0'
	print(time.time()-start)
	print(s)
	return s
psodwcnkey()