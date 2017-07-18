import datetime
import numpy as np

from hyperopt import hp, fmin, tpe
import os
import sys

from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.getcwd())

import workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QXgb2
from workdir.classes.models import qm




if __name__ == "__main__":
    CV_SCORE_TO_STOP = 0.544
    DATAS = [266, 269]

    EVALS_ROUNDS = 4000

    rounds = EVALS_ROUNDS

    cv = QCV(qm)
    counter = 0
    def fn(params):
        global counter

        counter +=1

        params['max_features'] = params['max_features'] / 10
        params['n_estimators'] = int(1.3 ** params['n_estimators'])

        model_id = qm.add_by_params(
            RandomForestClassifier(
                max_depth=int(params['max_depth']),
                n_estimators=int(params['n_estimators']),
                max_features=float(params['max_features']),
                n_jobs=-1
            ),
            'hyperopt rand_forest',
            predict_fn='predict_proba'
        )
        res = cv.cross_val(model_id, params['data_id'], seed=1000, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP)
        res = np.float64(res)
        res_arr = [res]
        # if res < CV_SCORE_TO_STOP:
        #     for i in range(7):
        #         res = cv.cross_val(model_id, data_id, seed=1001 + i, force=True)
        #         res = np.float64(res)
        #         res_arr.append(res)

        print(params['data_id'], model_id, "{}/{}".format(counter, rounds), res_arr, datetime.datetime.now(),  params)
        return np.mean(res_arr)
    space = {
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 18, 26, 1),
        'max_features': hp.quniform('max_features', 2, 10, 1)
    }

    counter = 0
    space['data_id'] = hp.choice('data_id', DATAS)
    rounds = EVALS_ROUNDS
    fmin(fn, space, algo=tpe.suggest, max_evals=rounds)
