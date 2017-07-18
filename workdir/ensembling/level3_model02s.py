import datetime
import random

import numpy as np

from hyperopt import hp, fmin, tpe

import os
import sys

from sklearn.linear_model import Ridge

sys.path.insert(0, os.getcwd())

import workdir.classes.config
from qml.cv import QCV
from qml.helpers import get_engine
from qml.models import QXgb, QAvg, QRankedAvg, QRankedByLineAvg, QStackModel
from workdir.classes.models import qm

if __name__ == "__main__":
    _, conn = get_engine()
    cv = QCV(qm)

    CV_SCORE_TO_SELECT = 0.53745
    CV_SCORE_TO_STOP = 0.542

    ROUNDS = 1000


    res = conn.execute(
        """
            select data_id, cls, descr, 
            substring_index(group_concat(model_id order by cv_score), ',', 20) as models
            from qml_results r 
            inner join qml_models m using(model_id) 
            where m.level=2 and cv_score < {}
            group by data_id, cls, descr
        """.format(CV_SCORE_TO_SELECT)
    ).fetchall()

    results = []

    for r in res:
        for m in r['models'].split(','):
            results.append([int(m), r['data_id'], 1000])


    for i in range(5000):
        random.shuffle(results)
        models = list(results[:random.randint(2, 20)])
        models = sorted(models, key=lambda x: (x[0], x[1]))
        print('{}/{}'.format(i, ROUNDS), models)

        model_id = qm.add_by_params(
            QAvg(models)
        )
        print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
        conn.execute("update qml_models set level=3 where model_id={}".format(model_id))

        # model_id = qm.add_by_params(
        #     QAvg(models, is_geom=True)
        # )
        # print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
        # conn.execute("update qml_models set level=3 where model_id={}".format(model_id))
        #
        # model_id = qm.add_by_params(
        #     QRankedAvg(models)
        # )
        # print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
        # conn.execute("update qml_models set level=3 where model_id={}".format(model_id))

        # model_id = qm.add_by_params(
        #     QRankedByLineAvg(models)
        # )
        # print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
        # conn.execute("update qml_models set level=3 where model_id={}".format(model_id))

        # if len(models) > 8:
        #     model_id2 = qm.add_by_params(
        #         Ridge(alpha=0.01),
        #     )
        #     model_id = qm.add_by_params(
        #         QStackModel(models, second_layer_model=model_id2, nsplits=2)
        #     )
        #     print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
        #     conn.execute("update qml_models set level=3 where model_id={}".format(model_id))
        #
        #
        #     model_id2 = qm.add_by_params(
        #         Ridge(alpha=0.05),
        #     )
        #     model_id = qm.add_by_params(
        #         QStackModel(models, second_layer_model=model_id2, nsplits=2)
        #     )
        #     print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
        #     conn.execute("update qml_models set level=3 where model_id={}".format(model_id))

    conn.execute(
        "update qml_models set level=3 where level=1 and cls in ('qavg', 'qrankedavg', 'QRankedByLineAvg', 'QStackModel')")
