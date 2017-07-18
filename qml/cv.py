import hashlib
from random import shuffle

import time

import sys

import logging

from hyperopt import hp, fmin, tpe
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


    def features_sel_del(self, model_id, data_id, early_stop_cv=None, exclude=None, splits=None, log_file=None):
        if log_file is not None:
            logging.basicConfig(filename=log_file, level=logging.INFO)
            log = lambda x: logging.info(x)
        else:
            log = lambda x: print(x)


        if exclude is None:
            exclude = []
        if type(data_id)==list:
            X, Y = data_id
            data_id = -1
        else:
            X = pd.read_csv(QML_TRAIN_X_FILE_MASK.format(data_id), index_col=QML_INDEX_COL)
            Y = pd.read_csv(QML_TRAIN_Y_FILE_MASK.format(data_id), index_col=QML_INDEX_COL)

        if splits is None:
            splits = self._load_splits(Y)

        results = []


        for col in set(X.columns) - set(exclude):
            X_curr = X.drop(labels=[col]+exclude, axis=1)

            total_score = self._features_sel_cv(X_curr, Y, splits, model_id, data_id, log, early_stop_cv)
            results.append([total_score, col])
            log("{0:0=3d}\t{1}\t{2}".format(len(exclude)+1, total_score, col))

        if set(X.columns) != set(exclude + [results[0][1]]):
            self.features_sel_del(model_id, [X, Y], early_stop_cv, exclude + [results[0][1]], splits, log_file=log_file)


    def features_sel_add(self, model_id, data_id, initial_cols, cols_to_add,  early_stop_cv=None, splits=None, log_file=None):
        if log_file is not None:
            logging.basicConfig(filename=log_file, level=logging.INFO)
            log = lambda x: logging.info(x)
        else:
            log = lambda x: print(x)

        if type(data_id)==list:
            X, Y = data_id
            data_id = -1
        else:
            X = pd.read_csv(QML_TRAIN_X_FILE_MASK.format(data_id), index_col=QML_INDEX_COL)
            Y = pd.read_csv(QML_TRAIN_Y_FILE_MASK.format(data_id), index_col=QML_INDEX_COL)

        if splits is None:
            splits = self._load_splits(Y)

        results = []
        for col in cols_to_add:
            X_curr = X[initial_cols+[col]]

            total_score = self._features_sel_cv(X_curr, Y, splits, model_id, data_id, log, early_stop_cv)
            results.append([total_score, col])
            log("{0:0=3d}\t{1}\t{2}".format(len(initial_cols), total_score, col))

        # if set(X.columns) != set(exclude + [results[0][1]]):
        #     self.features_sel_del(model_id, [X, Y], early_stop_cv, exclude + [results[0][1]], splits, log_file=log_file)



    def features_sel_hyper(self, model_id, data_id, early_stop_cv=None, log_file=None, rounds=10000):
        if log_file is not None:
            logging.basicConfig(filename=log_file, level=logging.INFO)
            log = lambda x: logging.info(x)
        else:
            log = lambda x: print(x)


        X = pd.read_csv(QML_TRAIN_X_FILE_MASK.format(data_id), index_col=QML_INDEX_COL)
        Y = pd.read_csv(QML_TRAIN_Y_FILE_MASK.format(data_id), index_col=QML_INDEX_COL)

        splits = self._load_splits(Y)

        def fn(params):
            cols = list([col[0] for col in params.items() if col[1]==1])
            log('   {}/{} {}'.format(len(cols), len(params.items()), sorted(cols)))
            X_curr = X[cols]
            total_score = self._features_sel_cv(X_curr, Y, splits, model_id, data_id, log, early_stop_cv)

            log("{}\t{}".format(total_score, sorted(cols)))

            return total_score

        space = {}
        for col in X.columns:
            space[col] = hp.choice(col, [0, 1])

        fmin(fn, space, algo=tpe.suggest, max_evals=rounds)


    def _features_sel_cv(self, X, Y, splits, model_id, data_id, log, early_stop_cv = None):

        #workaround to set first fold the worst, for using early stop cv
        splits_new_order_temp = []
        for train_indexes, test_indexes in splits:
            splits_new_order_temp += [[train_indexes, test_indexes]]

        splists_new_order = [splits_new_order_temp[2], splits_new_order_temp[1], splits_new_order_temp[3], splits_new_order_temp[0], splits_new_order_temp[4]]

        scores = []
        i = 0
        for train_indexes, test_indexes in splists_new_order:
            i += 1
            X_train = X.loc[train_indexes]
            Y_train = Y.loc[train_indexes][QML_RES_COL]
            X_test = X.loc[test_indexes]
            Y_test = Y.loc[test_indexes][QML_RES_COL]
            res = self.qm.qpredict(model_id, data_id, data=(X_train, Y_train, X_test), Y_test=Y_test, force=True,
                                   save_result=False)

            score = log_loss(Y_test, res.astype(np.float64), eps=1e-14)
            log('   {} {}'.format(i, score))
            sys.stdout.flush()
            scores.append(score)

            if early_stop_cv is not None:
                if early_stop_cv(score):
                    scores = [score]
                    break
        total_score = sum(scores) / len(scores)

        return total_score

    def cross_val(self, model_id, data_id, force=False, early_stop_cv = None, seed = 1000):
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

        splits = self._load_splits(Y)

        #workaround to set first fold the worst, for using early stop cv
        splits_new_order_temp = []
        for train_indexes, test_indexes in splits:
            splits_new_order_temp += [[train_indexes, test_indexes]]

        splists_new_order = [splits_new_order_temp[2], splits_new_order_temp[1], splits_new_order_temp[3], splits_new_order_temp[0], splits_new_order_temp[4]]

        scores = []
        i = 0
        for train_indexes, test_indexes in splists_new_order:
            i += 1
            if data_id>0:
                # only one data set used, so pass data
                X_train = X.loc[train_indexes]
                Y_train = Y.loc[train_indexes][QML_RES_COL]
                X_test = X.loc[test_indexes]
                Y_test = Y.loc[test_indexes][QML_RES_COL]
                res = self.qm.qpredict(model_id, data_id, data=(X_train, Y_train, X_test), force=force, seed=seed)
            else:
                # many datasets can be used, so pass only ids
                X_train = train_indexes
                X_test = test_indexes
                Y_test = Y.loc[test_indexes][QML_RES_COL]
                res = self.qm.qpredict(model_id, data_id, ids=[X_train, X_test], force=force, seed=seed)


            # fpr, tpr, thresholds = roc_curve(Y_test, res)
            # score = auc(fpr, tpr)
            score = log_loss(Y_test, res.astype(np.float64), eps=1e-14)
            print('   ', i, score)
            sys.stdout.flush()
            scores.append(score)

            self.conn.execute(
                """insert into
                        qml_results_statistic
                    set
                        cv_score={0}, data_id={1}, model_id={2}, fold={3}, seed={4}
                    on duplicate key update
                        cv_score = values(cv_score)
                """.format(
                    round(score, 10), data_id, model_id, i, seed
                )
            )


            if early_stop_cv is not None:
                if early_stop_cv(score):
                    scores = [score]
                    break

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
                round(total_score, 10), round(total_time, 1), data_id, model_id
            )
        )

        return total_score

    def _load_splits(self, Y):
        splits = load(self._get_splits_filename(self.nsplits, Y.index))
        if splits is None:
            splits = []
            kf = model_selection.KFold(n_splits=self.nsplits, shuffle=True)
            for ids1, ids2 in kf.split(Y):
                splits.append([sorted(Y.index[ids1]), sorted(Y.index[ids2])])
            save(splits, self._get_splits_filename(self.nsplits, Y.index))
        return splits

    def _get_splits_filename(self, nsplists, train_ids):
        """ all datasets with the same ids have the same splits"""
        ids_hash = hashlib.md5(
            '_'.join([str(i) for i in sorted(train_ids)]).encode('utf-8')
        ).hexdigest()
        filename = QML_DATA_DIR + 'cv_splits/splits_{}_trainlen{}__h_{}.csv'.format(
            nsplists, len(train_ids), ids_hash
        )
        return filename