import datetime
import random

import numpy as np

from hyperopt import hp, fmin, tpe

import os
import sys
sys.path.insert(0, os.getcwd())

import workdir.classes.config
from qml.cv import QCV
from qml.helpers import get_engine
from qml.models import QXgb, QAvg, QRankedAvg, QRankedByLineAvg, QStackModel
from workdir.classes.models import qm

if __name__ == "__main__":
    _, conn = get_engine()
    cv = QCV(qm)

    CV_SCORE_TO_SELECT = 0.56
    CV_SCORE_TO_STOP = 0.5416


    ROUNDS = 20000


    res = conn.execute(
        """
            select data_id, cls, descr, 
            substring_index(group_concat(model_id order by cv_score), ',', 5) as models
            from qml_results r 
            inner join qml_models m using(model_id) 
            where m.level=1 and cv_score < {} and data_id in (66, 69, 266, 269, 47 )
            group by data_id, cls, descr
        """.format(CV_SCORE_TO_SELECT)
    ).fetchall()

    results = []
    best_models = []

    for r in res:
        for m in r['models'].split(','):
            results.append([int(m), r['data_id']])


    for i in range(ROUNDS):
        random.shuffle(results)
        models = list(results[:random.randint(2, 10)])
        models = sorted(models, key=lambda x: (x[0], x[1]))
        print('{}/{}'.format(i, ROUNDS), models)


        try:
            model_id = qm.add_by_params(
                QAvg(models)
            )
            print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
            conn.execute("update qml_models set level=2 where model_id={}".format(model_id))

            # model_id = qm.add_by_params(
            #     QAvg(models, is_geom=True)
            # )
            # print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
            # conn.execute("update qml_models set level=2 where model_id={}".format(model_id))

            model_id = qm.add_by_params(
                QRankedAvg(models)
            )
            print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
            conn.execute("update qml_models set level=2 where model_id={}".format(model_id))

            model_id = qm.add_by_params(
                QRankedByLineAvg(models)
            )
            print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
            conn.execute("update qml_models set level=2 where model_id={}".format(model_id))

            # model_id = qm.add_by_params(
            #     QStackModel(models, second_layer_model=1)
            # )
            # print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
            # conn.execute("update qml_models set level=2 where model_id={}".format(model_id))
            #
            # model_id = qm.add_by_params(
            #     QStackModel(models, second_layer_model=2)
            # )
            # print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
            # conn.execute("update qml_models set level=2 where model_id={}".format(model_id))

            # if len(models) > 8:
            #     model_id = qm.add_by_params(
            #         QStackModel(models, second_layer_model=1747)
            #     )
            #     conn.execute("update qml_models set level=2 where model_id={}".format(model_id))
            #     print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x > CV_SCORE_TO_STOP))
            #
            #     model_id = qm.add_by_params(
            #         QStackModel(models, second_layer_model=1841)
            #     )
            #     conn.execute("update qml_models set level=2 where model_id={}".format(model_id))
            #     print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x > CV_SCORE_TO_STOP))


                # for result in results:
                #     model_id = qm.add_by_params(
                #         QStackModel(models, second_layer_model=result[0])
                #     )
                #     print(cv.cross_val(model_id, -1, early_stop_cv=lambda x: x>CV_SCORE_TO_STOP))
                #     conn.execute("update qml_models set level=2 where model_id={}".format(model_id))

        except:
            pass


    conn.execute(
        "update qml_models set level=2 where level=1 and cls in ('qavg', 'qrankedavg', 'QRankedByLineAvg', 'QStackModel')")
