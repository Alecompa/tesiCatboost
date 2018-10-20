import numpy as np
from catboost import CatBoostRegressor
import ROOT as rt
import math
import pickle
import matplotlib.pyplot as plt

def findsigma( preds ):
	h = rt.TH1F("h", "prova", 1200, 0 , 12 )
	for i in preds:
		h.Fill(i)
	#bin1 = h.FindFirstBinAbove(h.GetMaximum()/2)
	#bin2 = h.FindLastBinAbove(h.GetMaximum()/2)
	#sigma = h.GetBinCenter(bin2) - h.GetBinCenter(bin1)
	func = rt.TF1("prova","gaus")
	h.Fit(func, "Q", "0")
	sigma  = func.GetParameter(2)
	return sigma

def findmaximum( preds ):
	h = rt.TH1F("h", "prova", 1200, 0 , 12 )
	for i in preds:
		h.Fill(i)
	#maximum = h.GetBinCenter(h.GetMaximumBin())
	func = rt.TF1("prova","gaus")
	h.Fit(func, "Q", "0")
	maximum  = func.GetParameter(1)
	return maximum

maxPE = []
maxPEcoh = []
maxtime = []
sigmaPE = []
sigmaPEcoh = []
sigmatime = []

for e in range(10):

	#open dataset at fixed energy
	s = "dataset/eplus_"+str(e)+"MeV_simple.root"
	rf = rt.TFile(s,"read")
	evt_tree = rf.Get("evt")
	features = []

	#create features array
	for i, evt in enumerate(evt_tree):
		if (evt.r<17200) :
  			features.append([evt.totalPE_lpmt , evt.cohX, evt.cohY, evt.cohZ, evt.ht_mean - evt.ht1])

  	#load models
	modelPE = CatBoostRegressor()
	modelPE.load_model("models/totalPE_FV.model")
	modelPEcoh = CatBoostRegressor()
	modelPEcoh.load_model("models/totalPEcoh_FV.model")
	modeltime = CatBoostRegressor()
	modeltime.load_model("models/time_FV.model")

	#make predictions
	predictionsPE = modelPE.predict(features)
	predictionsPE += 0.511
	predictionsPEcoh = modelPEcoh.predict(features)
	predictionsPEcoh += 0.511
	predictionstime = modeltime.predict(features)
	predictionstime += 0.511
	
	#find prediction mean value and sigma
	sigmaPE.append(findsigma(predictionsPE))
	sigmaPEcoh.append(findsigma(predictionsPEcoh))
	sigmatime.append(findsigma(predictionstime))
	maxPE.append(findmaximum(predictionsPE))
	maxPEcoh.append(findmaximum(predictionsPEcoh))
	maxtime.append(findmaximum(predictionstime))
	
	rf.Close()

#plot data

y = []
y2 = []
y3 = []

for i in range(10):
	y.append(sigmatime[i]/(maxtime[i]))
	y2.append(sigmaPEcoh[i]/(maxPEcoh[i]))
	y3.append(sigmaPE[i]/(maxPE[i]))

#plt.style.use('ggplot')

t = np.arange(0.5, 10.0, 0.01)
s = np.sqrt( (2.85/np.sqrt(t))**2 + 0.9**2 )*0.01
plt.xlabel(r'$E_{vis}$')
plt.ylabel(r'$\sigma / E_{vis}$')
plt.grid(True)
plt.axis([0, 11, 0, 0.05])
plt.plot( maxtime , y, 'bs', label='nPE+coh+time' )
plt.plot( maxPEcoh, y2, 'rs', label='nPE+coh')
plt.plot( maxPE, y3, 'g^', label = 'nPE' )
plt.plot( t , s, '--', label = 'Other rec. alg.') 
plt.legend(loc='upper right')

plt.show()
