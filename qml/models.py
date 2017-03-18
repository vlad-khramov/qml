import inspect
import json
from collections import OrderedDict
from pathlib import Path
import random

import pandas as pd
import hashlib
import numpy as np
from scipy.stats.mstats import gmean
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from qml.cv import QCV
from qml.helpers import get_engine, save
from qml.config import *


class QModels:

    @classmethod
    def get_instance(cls):
        if not getattr(cls, 'instance', False):
            cls.instance = QModels()
        return cls.instance

    def __init__(self):
        self.models = {}
        #todo create cs in cv_data

    def qpredict(self, model_id, data_id, data=None, ids=None, tag='', save_result=True, save_model=False, force=False):
        if ids is not None:
            train_ids, test_ids = ids
            res = self._check_result_exists(model_id, data_id, train_ids, test_ids, tag)
            if res is not None and not force:
                return res

        if data is None:
            #todo
            load_data_id = data_id if data_id >0 else 1
            Y_train = pd.read_csv(QML_TRAIN_Y_FILE_MASK.format(load_data_id), index_col=QML_INDEX_COL)
            X_train = pd.read_csv(QML_TRAIN_X_FILE_MASK.format(load_data_id), index_col=QML_INDEX_COL)
            X_test = pd.read_csv(QML_TEST_X_FILE_MASK.format(load_data_id), index_col=QML_INDEX_COL)

            if ids is None:
                train_ids = Y_train.index.values
                test_ids = X_test.index.values
            else:
                temp = pd.concat([X_train, X_test])
                X_train = temp.loc[train_ids]
                Y_train = Y_train.loc[train_ids][QML_RES_COL]
                X_test = temp.loc[test_ids]
                temp = None
        else:
            X_train, Y_train, X_test = data
            train_ids = Y_train.index.values
            test_ids = X_test.index.values

        if ids is None and not force:
            res = self._check_result_exists(model_id, data_id, train_ids, test_ids, tag)
            if res is not None and not force:
                return res

        model = self.get_model(model_id)
        fit_res = model.fit(X_train, Y_train)

        if save_model:
            save(fit_res, QML_DATA_DIR + 'models/' + 'm{0:0=7d}_d{1:0=3d}__tr_{2}_ts_{3}_t_{4}'.format(
                model_id, data_id, len(X_train), len(X_test), tag
            ))

        predict_fn = getattr(fit_res, model.qml_predict_fn)

        predict_fn_kwargs = {}
        if 'force' in inspect.getfullargspec(predict_fn).args:
            predict_fn_kwargs['force'] = force


        res = predict_fn(X_test, **predict_fn_kwargs)


        #todo
        if model.qml_predict_fn == 'predict_proba':
            res = [x[1] for x in res]

        A1 = pd.DataFrame(X_test.index)
        A1[QML_RES_COL] = res
        A1.set_index(QML_INDEX_COL, inplace=True)
        if save_result:
            A1.to_csv(self._get_res_filename(model_id, data_id, train_ids, test_ids, tag))
        return A1

    def _check_result_exists(self, model_id, data_id, train_ids, test_ids, tag):
        """check if result already exists"""
        filename = self._get_res_filename(model_id, data_id, train_ids, test_ids, tag)
        my_file = Path(filename)
        if my_file.is_file():
            return pd.read_csv(filename, index_col=QML_INDEX_COL)
        else:
            return None

    def _get_res_filename(self, model_id, data_id, train_ids, test_ids, tag):
        """saved result filename"""
        ids_hash = hashlib.md5(
            '_'.join([str(i) for i in sorted(train_ids)]).encode('utf-8') +
            '_'.join([str(i) for i in sorted(test_ids)]).encode('utf-8')
        ).hexdigest()
        filename = QML_DATA_DIR + 'res/m{0:0=7d}_m{1:0=3d}__tr_{2}_ts_{3}_{4}_h_{5}.csv'.format(
            model_id, data_id, len(train_ids), len(test_ids), tag, ids_hash
        )
        return filename

    def get_model(self, model_id):
        if model_id not in self.models:
            self._load_model(model_id)
        return self.models[model_id]

    def add(self, model_id, model, description=None, predict_fn='predict', description_params=None):
        _, conn = get_engine()

        description = description if description else ''
        res = conn.execute("select cls, params, descr, predict_fn from qml_models where model_id={}".format(model_id)).fetchone()
        if res:
            if res['cls'] != self.get_class(model) or res['params'] != self.get_params(model, description_params):
                raise Exception('Model {} changed'.format(model_id))
        else:
            conn.execute(
                """
                    insert into qml_models (model_id, cls, params, descr, predict_fn) values
                    ({}, '{}', '{}', '{}', '{}')
                """.format(
                    model_id,
                    self.get_class(model),
                    self.get_params(model, description_params),
                    description,
                    predict_fn
                )
            )
        self.models[model_id] = model
        model.qml_descr = description
        model.qml_predict_fn = predict_fn

        conn.close()

    def add_by_params(self, model, description=None, predict_fn='predict', description_params=None):
        _, conn = get_engine()

        description = description if description else ''
        cls = self.get_class(model)
        description_params = self.get_params(model, description_params)

        res = conn.execute(
            """
                select model_id 
                from qml_models 
                where 
                    cls='{}'
                    and params='{}'
            """.format(cls, description_params)
        ).fetchone()
        if res:
            return res['model_id']
        else:
            conn.execute(
                """
                    insert into qml_models (model_id, cls, params, descr, predict_fn) values
                    (null, '{}', '{}', '{}', '{}')
                """.format(cls, description_params, description, predict_fn),
            )
        model_id=conn.execute('SELECT LAST_INSERT_ID() AS id').fetchone()[0]
        self.models[model_id] = model
        model.qml_descr = description
        model.qml_predict_fn = predict_fn

        conn.close()
        return model_id

    def get_class(self, model):
        return str(model.__class__.__name__)

    def get_params(self, model, description_params):
        return self.normalize_params(model.get_params()) if description_params is None else description_params

    @classmethod
    def normalize_params(cls, params):
        return json.dumps(OrderedDict(sorted(params.items())))

    def _load_model(self, model_id):
        _, conn = get_engine()

        #todo
        models = {
            'QXgb': QXgb,
            'KNeighborsClassifier': KNeighborsClassifier,
            'QAvg': QAvg,
            'QRankedAvg': QRankedAvg,
            'QRankedByLineAvg': QRankedByLineAvg,
            'QStackModel': QStackModel,
            'LogisticRegression': LogisticRegression,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'QPostProcessingModel': QPostProcessingModel,
            'RandomForestClassifier': RandomForestClassifier

        }

        res = conn.execute(
            """
                select cls, params, descr, predict_fn
                from qml_models 
                where 
                    model_id='{}'
            """.format(model_id)
        ).fetchone()

        if not res:
            raise Exception('Missing {} model'.format(model_id))

        model = models[res['cls']](**json.loads(res['params']))
        self.add(model_id, model, res['descr'], res['predict_fn'])



    def get_cv_score(self, model_id, data_id):
        _, conn = get_engine()

        row = conn.execute(
            "select cv_score from qml_results where model_id={} and data_id={}".format(model_id, data_id)
        ).fetchone()
        cv_score = row['cv_score']
        conn.close()

        return cv_score

####################################


class QXgb:
    def __init__(self, **kwargs):
        self.params = kwargs
        self._Booster = None

    def get_params(self):
        return self.params

    def fit(self, X, Y):
        trainDmatrix = xgb.DMatrix(X, label=Y)

        params = self.params.copy()
        params['silent'] = True
        train_params = {}
        train_params_names = ['num_boost_round', 'early_stopping_rounds']
        for n in train_params_names:
            if n in params:
                train_params[n] = params[n]
                del params[n]

        self._Booster = xgb.train(params, trainDmatrix, verbose_eval=False, **train_params)
        return self

    def predict(self, X):
        test_dmatrix = xgb.DMatrix(X)
        return self.booster().predict(test_dmatrix)


    ##from xgb##
    def booster(self):
        if self._Booster is None:
            raise Exception('need to call fit beforehand')
        return self._Booster

    @property
    def feature_importances_(self):
        """
        Returns
        -------
        feature_importances_ : array of shape = [n_features]

        """
        b = self.booster()
        fs = b.get_fscore()
        all_features = [fs.get(f, 0.) for f in b.feature_names]
        all_features = np.array(all_features, dtype=np.float32)
        return all_features / all_features.sum()

####################################

class QAvg:
    def __init__(self, models, is_geom=False):
        self.models = models
        self.is_geom = is_geom

    def get_params(self):
        p = {'models': self.models}
        if self.is_geom:
            p['is_geom'] = True
        return p

    def fit(self, X, Y):
        self.train_X, self.train_Y = X, Y
        return self

    def predict(self, X, force=False):
        qm = QModels.get_instance()

        A1 = None
        i=0
        for (model_id, data_id) in self.models:
            i+=1
            res = qm.qpredict(
                model_id, data_id, ids=[self.train_X.index.values, X.index.values], force=force
            )
            if A1 is None:
                A1 = res
            else:
                A1[QML_RES_COL + '{}'.format(i)] = res[QML_RES_COL]

        if self.is_geom:
            return list(gmean(A1, axis=1))
        else:
            return list(A1.mean(axis=1))


####################################

class QRankedAvg:
    def __init__(self, models):
        self.models = models

    def get_params(self):
        return {
            'models': self.models
        }

    def fit(self, X, Y):
        self.train_X, self.train_Y = X, Y
        return self

    def predict(self, X, force=False):
        qm = QModels.get_instance()

        A1 = None
        i=0
        cv_score_sum = 0
        for (model_id, data_id) in self.models:
            i+=1
            res = qm.qpredict(
                model_id, data_id, ids=[self.train_X.index.values, X.index.values], force=force
            )

            cv_score = qm.get_cv_score(model_id, data_id)
            cv_score_sum += cv_score

            if A1 is None:
                A1 = res
                A1['col{}'.format(i)] = A1[QML_RES_COL].apply(lambda x: float(x)*float(cv_score))
            else:
                A1['col{}'.format(i)] = res[QML_RES_COL].apply(lambda x: float(x)*float(cv_score))

        del A1[QML_RES_COL]
        return [float(x)/float(cv_score_sum) for x in list(A1.sum(axis=1))]

class QRankedByLineAvg:
    def __init__(self, models):
        self.models = models

    def get_params(self):
        return {
            'models': self.models
        }

    def fit(self, X, Y):
        self.train_X, self.train_Y = X, Y
        return self

    def predict(self, X, force=False):
        qm = QModels.get_instance()

        temp = []

        for (model_id, data_id) in self.models:
            res = qm.qpredict(
                model_id, data_id, ids=[self.train_X.index.values, X.index.values], force=force
            )

            cv_score = qm.get_cv_score(model_id, data_id)

            temp.append([model_id, data_id, cv_score, res])

        temp = sorted(temp, key=lambda x: x[2])
        A1 = None
        score_sum = 0
        for i, [model_id, data_id, cv_score, res] in enumerate(temp):
            score = i+1
            score_sum +=score
            if A1 is None:
                A1 = res
                A1['col{}'.format(i)] = A1[QML_RES_COL].apply(lambda x: float(x)*float(score))
            else:
                A1['col{}'.format(i)] = res[QML_RES_COL].apply(lambda x: float(x)*float(score))

        del A1[QML_RES_COL]
        return [float(x)/float(score_sum) for x in list(A1.sum(axis=1))]


####################################

class QStackModel:
    def __init__(self, models, second_layer_model):
        self.models = models
        self.second_layer_model = second_layer_model

    def get_params(self):
        return {
            'models': self.models,
            'second_layer_model': self.second_layer_model
        }

    def fit(self, X, Y):
        self.train_ids = list(X.index.values)
        self.train_Y = Y
        return self

    def predict(self, X, force=False):
        qm = QModels.get_instance()
        #cv = QCV(qm)

        ###level1
        middle = round(len(self.train_ids)/2)
        train_ids1 = self.train_ids[:middle]
        train_ids2 = self.train_ids[middle:]



        A_train = A_test = None
        for i, (model_id, data_id) in enumerate(self.models):
            res1 = qm.qpredict(
                model_id, data_id, ids=[train_ids1, train_ids2], force=force
            )
            res2 = qm.qpredict(
                model_id, data_id, ids=[train_ids2, train_ids1], force=force
            )

            col = pd.concat([res2, res1]).rename(index=str, columns={QML_RES_COL: "m_{}".format(i)})
            if A_train is None:
                A_train = col
            else:
                A_train = A_train.join(col)

            res = qm.qpredict(
                model_id, data_id, ids=[self.train_ids, X.index.values], force=force
            )

            col = res.rename(index=str, columns={QML_RES_COL: "m_{}".format(i)})
            if A_test is None:
                A_test = col
            else:
                A_test = A_test.join(col)


        return list(
            qm.qpredict(
                self.second_layer_model, -1, data=[A_train, self.train_Y, A_test],
                save_result=False
            )[QML_RES_COL]
        )

####################################

class QPostProcessingModel:
    def __init__(self, model_id, data_id, fn):
        self.model_id = model_id
        self.data_id = data_id
        self.fn = fn

    def get_params(self):
        return {
            'model_id': self.model_id,
            'data_id': self.data_id
        }

    def fit(self, X, Y):
        self.train_X = X
        self.train_Y = Y
        return self

    def predict(self, X, cv_parts='', cv_part=''):
        res = QModels.get_instance().qpredict(self.model_id, self.data_id, data=[self.train_X, self.train_Y, X])
        return self.fn(X, res)