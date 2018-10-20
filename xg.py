import numpy 
import xgboost as xgb
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
#  if i >= 100000: 
#    break
  features.append([evt.totalPE_lpmt, math.sqrt(evt.cohX**2 + evt.cohY**2), evt.cohZ, evt.ht_mean - evt.ht1])
  labels.append(evt.E0)

train_features = numpy.array(features[:950000])
test_features = numpy.array(features[950000:])
train_labels = labels[:950000]
test_labels = labels[950000:]

model = xgb.XGBRegressor( max_depth = 10, eta = 0.08)

#model.load_model("time.model")
model.fit(train_features, train_labels)
predictions = model.predict(test_features)
mse = mean_squared_error(test_labels, predictions)
print("MSE: %.4f" % mse)

model.save_model("prova.model")

#fig, ax = plt.subplots()
plt.hist2d(test_labels, predictions/test_labels, bins=(80,80),range=[[0,10],[0.60,1.30]], norm=LogNorm())
plt.colorbar()

plt.show()