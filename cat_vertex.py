import numpy as np
from catboost import CatBoostRegressor
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import ROOT as rt
import math
from sklearn.metrics import mean_squared_error

def findsigma( preds ):
	h = rt.TH1F("h", "prova", 100, -750 , 750 )
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
	h = rt.TH1F("h", "prova", 100, -750 , 750 )
	for i in preds:
		h.Fill(i)
	#maximum = h.GetBinCenter(h.GetMaximumBin())
	func = rt.TF1("prova","gaus")
	h.Fit(func, "Q", "0")
	maximum  = func.GetParameter(1)
	return maximum

rf = rt.TFile("dataset/eplus-flat_1M_simple.root","read")
evt_tree = rf.Get("evt")
features = []
labels = []
energy_true = []

for i, evt in enumerate(evt_tree):
	if (evt.r<17200) :	
		features.append([evt.totalPE_lpmt, evt.cohX, evt.cohY, evt.cohZ , evt.ht_mean - evt.ht1])
		labels.append(evt.r)
		energy_true.append(evt.E0)
print(len(features))

#train_features = np.array(features[:820000])
test_features = np.array(features[820000:])
#eval_features = features[770000:820000]
#train_labels = labels[:820000]
test_labels = labels[820000:]
#energy_test = energy_true[820000:]
#eval_labels = labels[770000:820000]

print(len(test_features))

model = CatBoostRegressor() #learning_rate = 0.1, iterations = 3000, depth=10, loss_function='RMSE')# l2_leaf_reg = 14, od_type = "Iter",od_wait = 50)

model.load_model("models/vertex.model")
#fit_model = model.fit(train_features, train_labels) #eval_set = (eval_features,eval_labels))
predictions = model.predict(test_features)

#print (fit_model.get_params())

#mse = mean_squared_error(test_labels, predictions)
#print("MSE: %.4f" % mse)

#model.save_model("vertex.model", format="cbm", export_parameters=None)

print(findmaximum(predictions-test_labels), findsigma(predictions-test_labels))
plt.hist(predictions- test_labels, bins=100,range=[-750,750])

#plt.hist2d(energy_test , predictions - test_labels, bins=(80,80),range=[[0,11],[-1000,1000]] , norm=LogNorm())

#plt.colorbar()


plt.show()