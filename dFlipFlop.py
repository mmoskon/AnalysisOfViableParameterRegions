import numpy as np 
import math
from math import pow
import numpy.fft as fft
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib import cm  
from scipy.integrate import odeint  

 	
'''
The deterministic model of biological D flip-flop in master slave cinfiguration
'''
class DFlipFlop: 
	
	def __init__(self, parameter_values, params, initial_conditions, threshold = -540000*1e3, dt = 0.01): #TO DO: popravi parameter viable points 
		#dictionary of form parameter: {min: val, max: val, ref: val}   
		self.nParams = len(params)  
		self.params = params #model parameters
		self.parameter_values = parameter_values #allowed parameter ranges  
		self.y0 = initial_conditions 
		self.threshold = threshold 	
		self.dt = dt
		self.T = 96 #in hours 
		self.N = int(self.T/self.dt) 
		self.ts = np.linspace(0, self.T, self.N)  
		self.per = self.T/4 
		self.amp = 100 #[nM]
		self.sample_rate 		= 0.0033333333 #Hz 
		self.samples_per_hour 	= (1/self.dt)
		self.jump = int(self.samples_per_hour/(self.sample_rate*3600))
		#clk signal
		self.CLK = [self.getClock(x) for x in np.linspace(0, self.T, self.N)]   
		#ideal response 
		self.nS = self.N/self.jump 
		self.ideal = [0]*self.N  
		self.ideal[0:int(self.N /4)] = [self.amp]*int(self.N /4) 
		self.ideal[2*int(self.N /4):3*int(self.N/4)] = [self.amp]*int(self.N /4)    						 		
		self.idealF = self.getFrequencies(self.ideal) 	
		self.threshold = -10*100  #10nM -+ from ideal signal harmonics, only first 10 are selected 
		self.modes = [self.eval] 
		
			
	def getClock(self, t):
		return self.amp*(np.sin(2*math.pi*(t)/self.per) + 1)/2

	def getFrequencies(self, y):
		#fft sample rate: 1 sample per 5 minutes
		y = y[0::self.jump]  
		res = abs(fft.rfft(y))
		#normalize the amplitudes 
		res = res/math.ceil(self.nS/2) 
		return res
		
	#evaluates a candidate 
	def eval(self, candidate):
		Y = np.array(self.simulate(candidate)) 		
		p1 = Y[:,2] #take q    
		fftData = self.getFrequencies(p1)   
		#take only first 10 harmonics  
		fftData = fftData[0:10] 
		idealF = self.idealF[0:10] 
		diff = fftData - idealF   
		cost = -np.dot(diff, diff)      		     		
		return cost,       
		
	def isViable(self, point): 	
		fitness = self.eval(point) 			
		return fitness[0] >= self.threshold  

	def getFrequencies(self, y):
		#fft sample rate: 1 sample per 5 minutes
		y = y[0::self.jump]  
		res = abs(fft.rfft(y))
		#normalize the amplitudes 
		res = res/math.ceil(self.nS/2) 
		return res 
		
	#simulates a candidate
	def simulate(self, candidate):		
		return odeint(self.flipFlopModelOde, self.y0, self.ts, args=(candidate,))  			 

	def getTotalVolume(self):
		vol = 1.0
		for param in self.params:		
			vol = vol*(self.parameter_values[param]["max"] - self.parameter_values[param]["min"])
		return vol

	def plotModel(self, subject): 
		ts = np.linspace(0, self.T, self.N)
		Y = self.simulate(subject) 			
		Y = np.array(Y) 
		p1 = self.CLK   
		p2 = Y[:,0]   
		p3 = Y[:,1] 
		p4 = Y[:,2] 
		p5 = Y[:,3] 

    
		lines = plt.plot(ts, p1, '#15A357', ts, p2, 'k', ts, p3, 'k--', ts, p4, '#A62B21', ts, p5, '#0E74C8')  
		plt.setp(lines[0], linewidth=1.5)
		plt.setp(lines[1], linewidth=1.5) 
		plt.setp(lines[2], linewidth=1.5)  
		plt.setp(lines[3], linewidth=1.5)  
		plt.setp(lines[4], linewidth=1.5)  
		
		plt.ylabel('Molarity [nM]') 
		plt.xlabel(r"Time [h]")   
		plt.legend(('$CLK$', '$a$', '$a_{c}$', '$q$', '$qc$'), loc='upper right')   		
		plt.show() 				
		
		
	def flipFlopModelOde(self, Y, t, can):  
		a     = Y.item(0) 
		not_a = Y.item(1)
		q     = Y.item(2)
		not_q = Y.item(3)
		d = not_q
		
		alpha1 	= can[0]
		alpha2 	= can[1]
		alpha3 	= can[2]
		alpha4 	= can[3]
		delta1 	= can[4]
		delta2 	= can[5]
		Kd 		= can[6]
		n 		= can[7] 
		
		clk = self.getClock(t) 
		
		
		try:
			da_dt     = alpha1*(pow(d/Kd, n)/(1 + pow(d/Kd, n) + pow(clk/Kd, n) + pow(d/Kd, n)*pow(clk/Kd, n))) + alpha2*(1/(1 + pow(not_a/Kd, n))) - delta1*a   	
			dnot_a_dt = alpha1*(1/(1 + pow(d/Kd, n) + pow(clk/Kd, n) + pow(d/Kd, n)*pow(clk/Kd, n))) + alpha2*(1/(1 + pow(a/Kd, n))) - delta1*not_a   
			dq_dt     = alpha3*((pow(a/Kd, n)*pow(clk/Kd, n))/(1 + pow(a/Kd, n) + pow(clk/Kd, n) + pow(a/Kd, n)*pow(clk/Kd, n))) + alpha4*(1/(1 + pow(not_q/Kd, n))) - delta2*q  
			dnot_q_dt = alpha3*((pow(not_a/Kd, n)*pow(clk/Kd, n))/(1 + pow(not_a/Kd, n) + pow(clk/Kd, n) + pow(not_a/Kd, n)*pow(clk/Kd, n))) + alpha4*(1/(1 + pow(q/Kd, n))) - delta2*not_q  
		except (OverflowError, ValueError):
			da_dt = 0
			dnot_a_dt = 0
			dq_dt = 0
			dnot_q_dt = 0 
		return np.array([da_dt, dnot_a_dt, dq_dt, dnot_q_dt]) 
	
	
	
	

