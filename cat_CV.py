import numpy 
from catboost import CatBoostRegressor
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import ROOT as rt
import math
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
#from pactools.grid_search import GridSearchCVProgressBar

rf = rt.TFile("dataset/eplus-flat_1M_simple.root","read")
evt_tree = rf.Get("evt")
features = []
labels = []

for i, evt in enumerate(evt_tree):
	if (evt.r<17200) :	
		features.append([evt.totalPE_lpmt, evt.cohX, evt.cohY, evt.cohZ, evt.ht_mean - evt.ht1])
		labels.append(evt.E0)

print(len(features))

train_features = numpy.array(features[:90000])
test_features = numpy.array(features[90000:100000])
#eval_features = features[770000:820000]
train_labels = labels[:90000]
test_labels = labels[90000:100000]
#eval_labels = labels[770000:820000]


params = {'depth': [ 4 , 7 , 10 ],
          'learning_rate' : [0.01, 0.03, 0.1],
          'l2_leaf_reg': [1, 4, 9] }
cb = CatBoostRegressor(loss_function='RMSE', iterations = 500, verbose = False )
cb_model = GridSearchCV(cb, params, scoring='neg_mean_squared_error', cv = 3, verbose = 100)
cb_model.fit(train_features, train_labels)

print("Best parameters set found on development set:")
print()
print(cb_model.best_params_)
print()
print("Grid scores on development set:")
print()
means = cb_model.cv_results_['mean_test_score']
stds = cb_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, cb_model.cv_results_['params']):
	print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

predictions = cb_model.predict(test_features)

# model.save_model("prova.model", format="cbm", export_parameters=None)

plt.hist2d(test_labels, predictions/test_labels, bins=(80,80),range=[[0,10],[0.60,1.30]] , norm=LogNorm())
plt.colorbar()

plt.show()


