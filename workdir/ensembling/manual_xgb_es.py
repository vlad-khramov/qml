import datetime
import numpy as np
import pandas as pd

import os
import sys

from sklearn import model_selection

sys.path.insert(0, os.getcwd())

from  workdir.classes.config import config
from qml.cv import QCV
from qml.models import QXgb
from workdir.classes.models import qm

cv = QCV(qm)

X = pd.read_csv(config.QML_TRAIN_X_FILE_MASK.format(66), index_col='id')
Y = pd.read_csv(config.QML_TRAIN_Y_FILE_MASK.format(66), index_col='id')
test = pd.read_csv(config.QML_TEST_X_FILE_MASK.format(66), index_col='id')
#all = pd.concat([train, test])

splits = []
kf = model_selection.ShuffleSplit(200, 0.05, random_state=1000)
for ids1, ids2 in kf.split(Y):
    splits.append([sorted(Y.index[ids1]), sorted(Y.index[ids2])])

#splits = cv._load_splits(Y)


for train_indexes, test_indexes in splits:
    X_train = X.loc[train_indexes]
    Y_train = Y.loc[train_indexes][config.QML_RES_COL]
    X_test = X.loc[test_indexes]
    Y_test = Y.loc[test_indexes][config.QML_RES_COL]

    model = qm._load_model(1747)
    model.params['lr_decay'] = 0.0001
    model.params['early_stopping_rounds'] = 100
    model.params['num_boost_round'] = 10000
    model.fit(X_train, Y_train, seed=1000, eval_set=[(X_test, Y_test)], verbose_eval=100)
    np.savetxt('workdir/ensembling/manual01/{}'.format(np.random.randint(0, 2000000000)), model.predict(test))


