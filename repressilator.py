import numpy as np 
import math
import peakutils
import numpy.fft as fft
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib import cm  
from scipy.integrate import odeint 
	
'''
The deterministic model of biological repressilator 
''' 
class Repressilator: 
	
	def __init__(self, parameter_values, params, initial_conditions, dt = 0.001, mode = 0): 
		self.nParams = len(params)   
		self.params = params #model parameters
		self.parameter_values = parameter_values #allowed parameter ranges  
		self.y0 = initial_conditions 	
		self.dt = dt
		self.T = 48 #hours
		self.N = int(self.T/self.dt) 
		self.ts = np.linspace(0, self.T, self.N) 
		self.amp = 300 #[nM] 		
		self.per = self.T/8 	
		self.sample_rate 		= 0.0033333333 #Hz 
		self.samples_per_hour 	= (1/self.dt)		
		self.jump 				= int(self.samples_per_hour/(self.sample_rate*3600))  	 	
		self.ideal = self.amp*(np.sin(math.pi*(self.ts)/self.per - math.pi/2) + 1) 
		#number of samples for FFT		
		self.nS = self.N/self.jump 
		self.dF = self.sample_rate/self.nS  
		self.idealF = self.getFrequencies(self.ideal) 		 	
		thresholdOne = -(self.nS/2)*100 #10nM -+ from ideal signal harmonics       
		thresholdTwo = 200  
		self.minAmp = 200
		self.maxAmp = 400 
		self.mode = mode    			
		self.modes = [self.eval]       
		self.threshold = thresholdOne  
		if self.mode == 1:
			self.threshold = thresholdTwo
	
	#gets sumed difderence of arrayData
	@staticmethod 	
	def getDif(indexes, arrayData):	
		arrLen = len(indexes)
		sum = 0
		for i, ind in enumerate(indexes):
			if i == arrLen - 1:
				break
			sum += arrayData[ind] - arrayData[indexes[i + 1]]
			
		#add last peak - same as substracting it from zero 
		sum += arrayData[indexes[-1:]]  
		return sum   
		
	#gets standard deviation 
	@staticmethod 
	def getSTD(indexes, arrayData, window):
		numPeaks = len(indexes)
		arrLen = len(arrayData)
		sum = 0
		for ind in indexes:
			minInd = max(0, ind - window)
			maxInd = min(arrLen, ind + window)
			sum += np.std(arrayData[minInd:maxInd])  
			
		sum = sum/numPeaks 	
		return sum	 
	
	def getFrequencies(self, y):
		#fft sample rate: 1 sample per 5 minutes
		y = y[0::self.jump]  
		res = abs(fft.rfft(y))
		#normalize the amplitudes 
		res = res/math.ceil(self.nS/2) 
		return res

	def costOne(self, Y): 
		p1 = Y[:,1]   
		fftData = self.getFrequencies(p1)     
		
		diff = fftData - self.idealF         
		cost = -np.dot(diff, diff) 		
		return cost,	
		
	def costTwo(self, Y, getAmplitude = False): 
		p1 = Y[:,1]  
		fftData = self.getFrequencies(p1)      
		fftData = np.array(fftData) 
		indexes = peakutils.indexes(fftData, thres=0.02/max(fftData), min_dist=1)  
		#in case of no oscillations return 0 
		if len(indexes) == 0:     
			return 0,  
		#if amplitude is greater than 400nM
		amp = np.max(fftData[indexes])
		if amp > self.maxAmp: 
			return 0, 
		fitSamples = fftData[indexes]  			
		std = self.getSTD(indexes, fftData, 1)  
		diff = self.getDif(indexes, fftData)  
		cost = std + diff
		#print(cost)   
		if getAmplitude:
			return cost, amp
		return cost, 
		
	def isViableFitness(self, fit):
		return fit >= self.threshold
		
	def isViable(self, point): 
		fitness = self.eval(point, getAmplitude=True)  
		if self.mode == 0:
			return self.isViableFitness(fitness[0]) 
			
		fit = fitness[0] 
		amp = 0
		if fit > 0:
			amp = fitness[1] 
		return self.isViableFitness(fit) and amp >= self.minAmp and amp <= self.maxAmp   
		
	#evaluates a candidate  
	def eval(self, candidate, getAmplitude = False): 
		Y = np.array(self.simulate(candidate)) 
		if self.mode == 0:
			return self.costOne(Y)  
		else:
			return self.costTwo(Y, getAmplitude)      
	
	#simulates a candidate
	def simulate(self, candidate): 
		return odeint(self.repressilatorModelOde, self.y0, self.ts, args=(candidate,))   		
		
	def plotModel(self, subject): 		
		ts = np.linspace(0, self.T, self.N)
		Y = self.simulate(subject) 			
		Y = np.array(Y) 
		
		p0 = Y[:,0] 
		p1 = Y[:,1]  
		p2 = Y[:,2]   
		p3 = Y[:,3]     
		p4 = Y[:,4] 
		p5 = Y[:,5] 
		
		lines = plt.plot(ts, p1, ts, p3, ts, p5)  
		plt.setp(lines[0], linewidth=1.5, c="#15A357")
		plt.setp(lines[1], linewidth=1.5, c="#0E74C8")
		plt.setp(lines[2], linewidth=1.5, c="#A62B21")     		 
		plt.ylabel('Molarity [nM]')  
		plt.xlabel(r"Time [h]")   
		plt.legend(('X', 'Y', 'Z'), loc='upper right')      		
		plt.show() 				

	def getTotalVolume(self):
		vol = 1.0
		for param in self.params:		
			vol = vol*(self.parameter_values[param]["max"] - self.parameter_values[param]["min"])
		return vol 

	def repressilatorModelOde(self, Y, t, can):  
		alpha = can[0]
		alpha0 = can[1]
		n = can[2]
		beta = can[3]
		deltaRNA = can[4] 
		deltaP = can[5]
		kd = can[6] 
		mx = Y.item(0)
		my = Y.item(2) 
		mz = Y.item(4)
		x = Y.item(1)
		y = Y.item(3) 
		z = Y.item(5) 
		
		#in case of math range error
		try:
			dmx = -deltaRNA*mx + alpha/(1 + math.pow(z/kd, n)) + alpha0 
			dmy = -deltaRNA*my + alpha/(1 + math.pow(x/kd, n)) + alpha0  
			dmz = -deltaRNA*mz + alpha/(1 + math.pow(y/kd, n)) + alpha0  
		except (OverflowError, ValueError):
			dmx = -deltaRNA*mx + alpha + alpha0
			dmy = -deltaRNA*my + alpha + alpha0
			dmz = -deltaRNA*mz + alpha + alpha0 
			
		dpx = beta*mx - deltaP*x
		dpy = beta*my - deltaP*y 
		dpz = beta*mz - deltaP*z
		
		return np.array([dmx, dpx, dmy, dpy, dmz, dpz])
	
	
	
	

