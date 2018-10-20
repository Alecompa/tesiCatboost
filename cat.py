import numpy 
from catboost import CatBoostRegressor
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import ROOT as rt
import math
from sklearn.metrics import mean_squared_error

rf = rt.TFile("dataset/eplus-flat_1M_simple.root","read")
evt_tree = rf.Get("evt")
features = []
labels = []

for i, evt in enumerate(evt_tree):
	if (evt.r<17200) :	
		features.append([evt.totalPE_lpmt, evt.cohX, evt.cohY, evt.cohZ ]) # evt.ht_mean - evt.ht1])
		labels.append(evt.E0)

print(len(features))

train_features = numpy.array(features[:820000])
test_features = numpy.array(features[820000:])
#eval_features = features[770000:820000]
train_labels = labels[:820000]
test_labels = labels[820000:]
#eval_labels = labels[770000:820000]

model = CatBoostRegressor()  #learning_rate = 0.1, iterations = 500, depth=10, loss_function='RMSE', l2_leaf_reg = 14, od_type = "Iter",od_wait = 50)

#model.load_model("time.model")
fit_model = model.fit(train_features, train_labels) #eval_set = (eval_features,eval_labels))
predictions = model.predict(test_features)

print (fit_model.get_params())

mse = mean_squared_error(test_labels, predictions)
print("MSE: %.4f" % mse)

model.save_model("prova.model", format="cbm", export_parameters=None)

#fig, ax = plt.subplots()
plt.hist2d(test_labels, predictions/test_labels, bins=(80,80),range=[[0,10],[0.60,1.30]] , norm=LogNorm())
plt.colorbar()

plt.show()