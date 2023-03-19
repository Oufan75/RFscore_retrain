###########################################################
# RF-Score_pred.r (all rights reserved)					  #
# 	Author: Dr Pedro J. Ballester                         #
#	Usage:  Read Appendix_A1.doc  		                  #
###########################################################
# Rewrite of RF-Score_pred.r in python

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Reading Training (TRN) and Test (TST) Datasets; generated with RFscore_descriptor.py
traindata = np.load("PDBbind_refined07_desc.npz")
testdata = np.load("PDBbind_core07_desc.npz")

ntrndata = len(traindata["pdbid"])  # number of pdb complexes for training
ntstdata = len(testdata["pdbid"])  # number of pdb complexes for testing
seed = 1

itrain = np.arange(ntrndata)
nsample = ntrndata
np.random.seed(seed)
np.random.shuffle(itrain)  # shuffle selected complexes
train_y = np.array(traindata["affinity"])[itrain]
train_X = np.reshape(traindata["feature"], (ntrndata, -1))[itrain]
train_pdbnames = np.array(traindata["pdbid"])[itrain]
test_y = np.array(testdata["affinity"])
test_X = np.reshape(testdata["feature"], (ntstdata, -1))
test_pdbnames = testdata["pdbid"]

# Data pre-processing; remove all zeros entries
train_X_f = np.reshape(train_X, (ntrndata, -1))
test_X_f = np.reshape(test_X, (ntstdata, -1))
all_X = np.vstack((train_X_f, test_X_f))
col_mask = np.sum(all_X, axis = 0) > 0
#col_mask = np.r_[1:3, 6, 10:12, 15, 19:21, 24, 28:30, 33, 37:39, 42, 
#                 46:48, 51, 55:57, 60, 64:66, 69, 73:75, 78]
# 36 features noted in the paper
train_Xs = train_X_f[:, col_mask]
test_Xs = test_X_f[:, col_mask]

# Selecting RF with best internal validation (RF-SCORE)
rmse_OOB_best = 1e8  # dummy high value
for mtry in range(2, col_mask.sum() + 1):
    RF_mtry = RandomForestRegressor(n_estimators=500, max_features=mtry, oob_score=True)
    RF_mtry.fit(train_Xs, train_y)
    rmse_OOB = np.sqrt(np.mean((RF_mtry.oob_prediction_ - train_y) ** 2))
    if rmse_OOB < rmse_OOB_best:
        mbest = mtry
        rmse_OOB_best = rmse_OOB
        print("mbest = ", mbest, "rmse_OOB = ", round(rmse_OOB, 3))
    print("mtry = ", mtry)

RF_Score = RandomForestRegressor(n_estimators=500, max_features=mbest)
RF_Score.fit(train_Xs, train_y)
train_pred = RF_Score.predict(train_Xs)
train_rmse = np.round(((train_y - train_pred)**2).mean() ** 0.5, 3)
train_sdev = np.round((train_y - train_pred).std(), 3)
fitpoints = pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_pred)], axis=1).dropna()
fitcorr = np.round(fitpoints.corr(method='pearson').iloc[0, 1], 3)
sprcorr = np.round(fitpoints.corr(method='spearman').iloc[0, 1], 3)

test_pred = RF_Score.predict(test_Xs)
test_rmse = np.round(((test_y - test_pred)**2).mean() ** 0.5, 3)
test_sdev = np.round((test_y - test_pred).std(), 3)
fitpoints = pd.concat([pd.DataFrame(test_y), pd.DataFrame(test_pred)], axis=1).dropna()
fitcorr = np.round(fitpoints.corr(method='pearson').iloc[0, 1], 3)
sprcorr = np.round(fitpoints.corr(method='spearman').iloc[0, 1], 3)


def plot_prediction(fitpoints, mode="train"):
    plt.plot(fitpoints.iloc[:,0], fitpoints.iloc[:,1], 'o')
    prline = np.polyfit(fitpoints.iloc[:,0], fitpoints.iloc[:,1], 1)
    plt.plot(fitpoints.iloc[:,0], np.polyval(prline, fitpoints.iloc[:,0]), '-')
    plt.title("R=" + str(fitcorr) + f" on {mode} set (" + str(fitpoints.shape[0]) + " complexes)")
    plt.xlabel("Measured binding affinity (PDBbind DB)")
    plt.ylabel("Predicted binding affinity (RF-Score)")
    plt.grid()
    plt.show()

# Variable importance by RF-SCORE
def plot_importance(RF_Score, X, pdbnames):
    importances = RF_Score.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Variable Importance")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), pdbnames[indices], rotation=90)
    plt.show()


