import datetime
import numpy as np

from hyperopt import hp, fmin, tpe
import os
import sys
sys.path.insert(0, os.getcwd())
import workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QAvg, QAvgOneModelData
from workdir.classes.models import qm



cv = QCV(qm)

model_id = qm.add_by_params(
    QXgb(
** {"alpha": 1.0, "booster": "gbtree", "colsample_bylevel": 0.7, "colsample_bytree": 0.8, "eta": 0.004, "eval_metric": "logloss",
    "gamma": 0.2, "max_depth": 4, "num_boost_round": 2015, "objective": "binary:logistic", "subsample": 0.8, "tree_method": "hist"}
    ),
    'hyperopt xgb', level=-1
)

model_id =qm.add_by_params(QAvgOneModelData(model_id, 3), level=-2)

cv.features_sel_del(model_id, 66, early_stop_cv=lambda x: x>0.5414, log_file='workdir/logs/feat19.txt', exclude=[])


