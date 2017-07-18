import datetime
import numpy as np

from hyperopt import hp, fmin, tpe
import os
import sys

from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.getcwd())

import workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QXgb2, QNN1, QNN2
from workdir.classes.models import qm




if __name__ == "__main__":
    CV_SCORE_TO_STOP = 0.548
    DATAS = [266, 269]

    EVALS_ROUNDS = 4000

    rounds = EVALS_ROUNDS

    cv = QCV(qm)
    counter = 0
    def fn(params):
        global counter

        counter +=1

        params['epochs'] = int(1.3 ** params['epochs'])

        model_id = qm.add_by_params(
            QNN2(
                middle_dim=int(params['middle_dim']),
                epochs=int(params['epochs'])
            ),
            'hyperopt nn1'
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
        'middle_dim': hp.quniform('middle_dim', 50, 250, 1),
        'epochs': hp.quniform('epochs', 18, 23, 1),
    }

    counter = 0
    space['data_id'] = hp.choice('data_id', DATAS)
    rounds = EVALS_ROUNDS
    fmin(fn, space, algo=tpe.suggest, max_evals=rounds)
