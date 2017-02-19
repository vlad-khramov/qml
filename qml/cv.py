from random import shuffle

import time
from sklearn import model_selection
from sklearn.metrics import auc

from sklearn.metrics import roc_curve, log_loss
from qml.config import *
from qml.helpers import get_engine, load, save

import pandas as pd


class QCV:
    def __init__(self, qmodels):
        _, self.conn = get_engine()
        self.qm = qmodels
        self.nsplits = 5

        self.X_train_cache = {}

        self.Y_train = pd.read_csv(QML_TRAIN_Y_FILE, index_col=QML_INDEX_COL)

        self.splits = load(QML_DATA_DIR + 'cv_splits_{}_ids.txt'.format(self.nsplits))
        if self.splits is None:
            kf = model_selection.KFold(n_splits=self.nsplits, shuffle=True)
            self.splits = []
            for ids1, ids2 in kf.split(self.Y_train):
                self.splits.append([sorted(self.Y_train.index[ids1]), sorted(self.Y_train.index[ids2])])
            save(self.splits, QML_DATA_DIR + 'cv_splits_{}_ids.txt'.format(self.nsplits))

    def cross_val(self, model_id, data_id, force=False, avg=False):
        start_time = time.time()

        rows = self.conn.execute(
            "select cv_score, cv_time from qml_results where data_id={} and model_id={}".format(data_id, model_id)
        ).fetchone()
        if rows and rows['cv_score'] is not None and not force:
            return rows['cv_score']

        if data_id in self.X_train_cache:
            X = self.X_train_cache[data_id]
        else:
            X = pd.read_csv(QML_TRAIN_X_FILE_MASK.format(data_id), index_col=QML_INDEX_COL)
            self.X_train_cache[data_id] = X

        Y = self.Y_train
        splits = self.splits

        scores = []
        i = 0
        for train_indexes, test_indexes in splits:
            i += 1
            if avg:
                X_train = train_indexes
                X_test = test_indexes
                Y_test = Y.loc[test_indexes][QML_RES_COL]
                res = self.qm.qpredict(model_id, data_id, ids=(X_train, X_test), cv_parts=self.nsplits, cv_part=i)
            else:
                X_train = X.loc[train_indexes]
                Y_train = Y.loc[train_indexes][QML_RES_COL]
                X_test = X.loc[test_indexes]
                Y_test = Y.loc[test_indexes][QML_RES_COL]
                res = self.qm.qpredict(model_id, data_id, data=(X_train, Y_train, X_test), cv_parts=self.nsplits, cv_part=i)

            fpr, tpr, thresholds = log_loss(Y_test, res) #roc_curve
            score = auc(fpr, tpr)

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
                round(total_score, 5), round(total_time, 1), data_id, model_id
            )
        )

        return total_score
