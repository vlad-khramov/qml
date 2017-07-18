import datetime
import numpy as np

from hyperopt import hp, fmin, tpe
import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.getcwd())

import workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QXgb2, QStackModel
from workdir.classes.models import qm




if __name__ == "__main__":
    CV_SCORE_TO_STOP = 0.5413
    DATAS = [219, 266, 269]

    EVALS_ROUNDS = 4000

    rounds = EVALS_ROUNDS

    cv = QCV(qm)
    counter = 0
    def fn(params):
        global counter

        counter +=1

        params['C'] = params['C'] / 10
        params['max_iter'] = int(1.3 ** params['max_iter'])

        model_id = qm.add_by_params(
            LogisticRegression(
                n_jobs=-1,
                max_iter=params['max_iter'],
                solver=params['solver'],
                C=params['C']
            ),
            'hyperopt log_regr',
            predict_fn='predict_proba'
        )
        model_id_main = qm.add_by_params(
            QStackModel(
                [
                    [1747, 69, 1000],
                    [101655, 266, 1000],
                    [1831, 266, 1000],
                    [101457, 266, 1000]


                    # [1747, 69, 1000],
                    # [1747, 66, 1000],
                    # [1747, 47, 1000],
                    #[1747, 69, 1001],
                    # [101340, 69, 1000],
                    # [101340, 66, 1000],
                    # [101340, 47, 1000],
                    # #[101340, 69, 1001],
                    # [101331, 69, 1000],
                    # [101331, 66, 1000],
                    # [101331, 47, 1000],
                    # #[101331, 69, 1001],
                    # [101261, 69, 1000],
                    # [101261, 66, 1000],
                    # [101261, 47, 1000],
                    #
                    # [101655, 266, 1000],
                    # [101655, 269, 1000],
                    #
                    # [1831, 266, 1000],
                    # [1831, 269, 1000],
                    #
                    # [101457, 266, 1000],
                    # [101457, 269, 1000],
                    #
                    # [101657, 269, 1000],
                    # [101657, 266, 1000],
                    #
                    # [1841, 269, 1000],
                    # [1841, 266, 1000],
                    #
                    # [101411, 269, 1000],
                    # [101411, 266, 1000],
                ],
                second_layer_model=model_id,
                nsplits=5
            ), level=-2
        )

        res = cv.cross_val(model_id_main, params['data_id'], seed=1000, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP)
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
        'max_iter': hp.quniform('max_iter', 18, 28, 1),
        'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag']),
        'C': hp.quniform('C', 5, 40,1),
    }

    counter = 0
    space['data_id'] = hp.choice('data_id', DATAS)
    rounds = EVALS_ROUNDS
    fmin(fn, space, algo=tpe.suggest, max_evals=rounds)
