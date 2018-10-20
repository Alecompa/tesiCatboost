import numpy as np
from catboost import CatBoostRegressor
import ROOT as rt
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats



#open dataset at fixed energy
s = "dataset/eplus-flat_1M_simple.root"
rf = rt.TFile(s,"read")
evt_tree = rf.Get("evt")
features = []
energies = []
radius = []

#create features array
for i, evt in enumerate(evt_tree):
	if (evt.r<17200) :
		features.append([evt.totalPE_lpmt , evt.cohX, evt.cohY, evt.cohZ, evt.ht_mean - evt.ht1])
		energies.append(evt.E0+0.511)
		#radius.append(evt.r)

test_dts = features[820000:]
labels = energies[820000:]

#load models
model = CatBoostRegressor()
model.load_model("models/time_FV.model")


#make predictions
predictions = model.predict(test_dts)
predictions += 0.511

print(np.mean(predictions))
h = []
#outliners = []

for i in range(len(predictions)):
	h.append((predictions[i]-labels[i])/labels[i])
	#if (predictions[i]-energies[i])/energies[i] < (-0.2) :
	#	outliners.append(radius[i])

#print(outliners)
#plt.plot(labels,h, 'b.')
#plt.yscale('log', nonposy='clip')
#plt.grid(True)

#plt.show()

sns.jointplot(labels, h, kind="kde");