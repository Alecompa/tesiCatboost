import numpy as np
import xgboost as xgb
import ROOT as rt
import math
import pickle
import matplotlib.pyplot as plt

def findfwhm( preds ):
	h = rt.TH1F("h", "prova", 800, 0 , 12 )
	for i in preds:
		h.Fill(i)
	bin1 = h.FindFirstBinAbove(h.GetMaximum()/2)
	bin2 = h.FindLastBinAbove(h.GetMaximum()/2)
	fwhm = h.GetBinCenter(bin2) - h.GetBinCenter(bin1)
	return fwhm

def findmaximum( preds ):
	h = rt.TH1F("h", "prova", 800, 0 , 12 )
	for i in preds:
		h.Fill(i)
	maximum = h.GetBinCenter(h.GetMaximumBin())
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
  		features.append([evt.totalPE_lpmt , math.sqrt(evt.cohX**2 + evt.cohY**2), evt.cohZ, evt.ht_mean - evt.ht1])

  	#load models
	modelPE = xgb.XGBRegressor()
	modelPE.load_model("models/totalPE_xg.model")
	modelPEcoh = xgb.XGBRegressor()
	modelPEcoh.load_model("models/totalPEcoh_xg.model")
	modeltime = xgb.XGBRegressor()
	modeltime.load_model("models/time_xg.model")

	#make predictions
	predictionsPE = modelPE.predict(features)
	predictionsPE += 0.511
	predictionsPEcoh = modelPEcoh.predict(features)
	predictionsPEcoh += 0.511
	predictionstime = modeltime.predict(features)
	predictionstime += 0.511
	
	#find prediction mean value and sigma
	sigmaPE.append(findfwhm(predictionsPE))
	sigmaPEcoh.append(findfwhm(predictionsPEcoh))
	sigmatime.append(findfwhm(predictionstime))
	maxPE.append(findmaximum(predictionsPE))
	maxPEcoh.append(findmaximum(predictionsPEcoh))
	maxtime.append(findmaximum(predictionstime))
	
	rf.Close()

#plot data

y = []
y2 = []
y3 = []

for i in range(10):
	y.append(sigmatime[i]/maxtime[i])
	y2.append(sigmaPEcoh[i]/maxPEcoh[i])
	y3.append(sigmaPE[i]/maxPE[i])

t = np.arange(0.25, 10.0, 0.01)
s = np.sqrt( (2.85/t)**2 + 0.9**2 )*0.01
plt.xlabel(r'$E_{vis}$')
plt.ylabel(r'$\sigma / E_{vis}$')
plt.grid(True)
plt.plot( maxtime , y, 'bs', label='nPE+coh+time' )
plt.plot( maxPEcoh, y2, 'rs', label='nPE+coh')
plt.plot( maxPE, y3, 'g^', label = 'nPE' )
plt.plot( t , s, '--', label = 'Other rec. alg.') 
plt.legend(loc='upper right')
plt.show()
