import hashlib
from random import shuffle

import time

import sys
from sklearn import model_selection
from sklearn.metrics import auc

from sklearn.metrics import roc_curve, log_loss
import numpy as np
from qml.config import *
from qml.helpers import get_engine, load, save

import pandas as pd


class QCV:
    def __init__(self, qmodels):
        _, self.conn = get_engine()
        self.qm = qmodels
        self.nsplits = 5

        self.X_train_cache = {}
        self.Y_train_cache = {}

    def cross_val(self, model_id, data_id, force=False):
        start_time = time.time()

        if not force:
            rows = self.conn.execute(
                "select cv_score, cv_time from qml_results where data_id={} and model_id={}".format(data_id, model_id)
            ).fetchone()
            if rows and rows['cv_score'] is not None:
                return rows['cv_score']

        if data_id > 0:
            if data_id in self.X_train_cache:
                X = self.X_train_cache[data_id]
            else:
                X = pd.read_csv(QML_TRAIN_X_FILE_MASK.format(data_id), index_col=QML_INDEX_COL)
                self.X_train_cache[data_id] = X

            if data_id in self.Y_train_cache:
                Y = self.Y_train_cache[data_id]
            else:
                Y = pd.read_csv(QML_TRAIN_Y_FILE_MASK.format(data_id), index_col=QML_INDEX_COL)
                self.Y_train_cache[data_id] = Y
        else:
            #todo
            if 1 in self.Y_train_cache:
                Y = self.Y_train_cache[1]
            else:
                Y = pd.read_csv(QML_TRAIN_Y_FILE_MASK.format(1), index_col=QML_INDEX_COL)
                self.Y_train_cache[1] = Y


        splits = load(self._get_splits_filename(self.nsplits, Y.index))
        if splits is None:
            splits = []
            kf = model_selection.KFold(n_splits=self.nsplits, shuffle=True)
            for ids1, ids2 in kf.split(Y):
                splits.append([sorted(Y.index[ids1]), sorted(Y.index[ids2])])
            save(splits, self._get_splits_filename(self.nsplits, Y.index))



        scores = []
        i = 0
        for train_indexes, test_indexes in splits:
            i += 1
            if data_id>0:
                # only one data set used, so pass data
                X_train = X.loc[train_indexes]
                Y_train = Y.loc[train_indexes][QML_RES_COL]
                X_test = X.loc[test_indexes]
                Y_test = Y.loc[test_indexes][QML_RES_COL]
                res = self.qm.qpredict(model_id, data_id, data=(X_train, Y_train, X_test), force=force)
            else:
                # many datasets can be used, so pass only ids
                X_train = train_indexes
                X_test = test_indexes
                Y_test = Y.loc[test_indexes][QML_RES_COL]
                res = self.qm.qpredict(model_id, data_id, ids=[X_train, X_test], force=force)


            # fpr, tpr, thresholds = roc_curve(Y_test, res)
            # score = auc(fpr, tpr)
            score = log_loss(Y_test, res.astype(np.float64), eps=1e-14)
            print(i, score)
            sys.stdout.flush()
            scores.append(score)

        total_score = sum(scores) / len(scores)
        total_time = time.time() - start_time

        self.conn.execute(
            """insert into
                    qml_results
                set
                    cv_score={0}, cv_time={1}, data_id={2}, model_id={3}
                on duplicate key update
                    cv_score = values(cv_score),
                    cv_time = values(cv_time)
            """.format(
                round(total_score, 6), round(total_time, 1), data_id, model_id
            )
        )

        return total_score


    def _get_splits_filename(self, nsplists, train_ids):
        """ all datasets with the same ids have the same splits"""
        ids_hash = hashlib.md5(
            '_'.join([str(i) for i in sorted(train_ids)]).encode('utf-8')
        ).hexdigest()
        filename = QML_DATA_DIR + 'cv_splits/splits_{}_trainlen{}__h_{}.csv'.format(
            nsplists, len(train_ids), ids_hash
        )
        return filename