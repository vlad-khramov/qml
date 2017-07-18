import datetime
import numpy as np

from hyperopt import hp, fmin, tpe
import os
import sys
sys.path.insert(0, os.getcwd())

import workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QStackModel
from workdir.classes.models import qm




if __name__ == "__main__":
    CV_SCORE_TO_STOP = 0.5412
    DATAS = [66]

    EVALS_ROUNDS = 100000

    rounds = EVALS_ROUNDS

    cv = QCV(qm)
    counter = 0
    def fn(params):
        global counter


        counter +=1
        data_id = params['data_id']
        del params['data_id']
        params['num_boost_rounds'] = int(1.3**params['num_boost_rounds'])
        params['eta'] = round(1 / (1.3**params['eta']), 4)
        params['subsample'] = 1#params['subsample']/10
        params['colsample_bytree'] = 1#params['colsample_bytree']/10
        params['colsample_bylevel'] = 1#params['colsample_bylevel']/10

        params['gamma'] = round(5 ** params['gamma'], 3)
        params['alpha'] = round(5 ** params['alpha'], 3)
        model_id = qm.add_by_params(
            QXgb(
                booster='gbtree',
                objective='binary:logistic',
                eval_metric='logloss',
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                colsample_bylevel=params['colsample_bylevel'],
                eta=params['eta'],
                gamma=params['gamma'],
                alpha=params['alpha'],
                max_depth=params['maxdepth'],
                num_boost_round=params['num_boost_rounds']#,		tree_method='hist'
            ),
            'hyperopt xgb'
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


        res = cv.cross_val(model_id_main, data_id, seed=1000, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP)
        res = np.float64(res)
        res_arr = [res]
        # if res < CV_SCORE_TO_STOP:
        #     for i in range(7):
        #         res = cv.cross_val(model_id, data_id, seed=1001 + i, force=True)
        #         res = np.float64(res)
        #         res_arr.append(res)

        print(data_id, model_id, "{}/{}".format(counter, rounds), res_arr, datetime.datetime.now(),  params)
        return np.mean(res_arr)
    space = {
        'subsample': hp.quniform('subsample', 4, 10, 1),
        'colsample_bytree': hp.quniform('colsample_bytree', 4, 10, 1),
        'colsample_bylevel': hp.quniform('colsample_bylevel', 4, 10, 1),
        'gamma': hp.quniform('gamma', -5, 0, 1),
        'alpha': hp.quniform('alpha', -5, 0, 1),
        'eta': hp.quniform('eta', 0, 34, 1),
        'maxdepth': hp.choice('maxdepth', range(1, 6)),
        'num_boost_rounds': hp.quniform('num_boost_rounds', 18, 30, 1)
    }

    for data_id in DATAS:
        counter = 0
        space['data_id'] = hp.choice('data_id', [data_id])
        rounds = EVALS_ROUNDS
        fmin(fn, space, algo=tpe.suggest, max_evals=rounds)
