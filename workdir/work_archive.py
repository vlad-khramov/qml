


def fn(params):
    params['kn'] = int(params['kn'])
    params['leaf_size'] = int(params['leaf_size'])
    model_id = qm.add_by_params(
        KNeighborsClassifier(params['kn'], leaf_size=params['leaf_size']),
        'hyperopt kn',
        predict_fn='predict_proba'
    )
    res = cv.cross_val(model_id, params['data'])
    print(datetime.datetime.now(), res, model_id, params)
    return res
space = {
    'kn': hp.quniform('hp', 1, 100, 1),
    'leaf_size': hp.quniform('leaf_size', 16, 64, 1),
    'data': hp.choice('data', [1, 2, 3]),
}
best = fmin(fn, space, algo=tpe.suggest, max_evals=100)
print(best)



def fn(params):
    params['num_boost_rounds'] = int(1.2**params['num_boost_rounds'])
    params['eta'] = round(1 / (1.2**params['eta']), 4)
    params['subsample'] = params['subsample']/10
    model_id = qm.add_by_params(
        QXgb(
            booster='gbtree',
            objective='binary:logistic',
            eval_metric='logloss',
            subsample=params['subsample'],
            eta=params['eta'],
            max_depth=params['maxdepth'],
            num_boost_round=params['num_boost_rounds']
        ),
        'hyperopt xgb'
    )
    res = cv.cross_val(model_id, 2)
    print(datetime.datetime.now(), res, model_id, params)
    return res
space = {
    'subsample': hp.quniform('subsample', 1, 10, 1),
    'eta': hp.uniform('eta', 0, 35),
    'maxdepth': hp.choice('maxdepth', range(1, 10)),
    'num_boost_rounds': hp.uniform('num_boost_rounds', 10, 36)
}
best = fmin(fn, space, algo=tpe.suggest, max_evals=500)
print(best)



########################

if __name__ == '__main__':
    cv = QCV(qm)

    def fn(params):
        params['max_iter'] = int(params['max_iter'])
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
        res1 = cv.cross_val(model_id, 1)
        print(datetime.datetime.now(), res1, model_id, params)
        res2 = cv.cross_val(model_id, 2)
        print(datetime.datetime.now(), res2, model_id, params)
        res3 = cv.cross_val(model_id, 4)
        print(datetime.datetime.now(), res1, model_id, params)
        res4 = cv.cross_val(model_id, 5)
        print(datetime.datetime.now(), res2, model_id, params)
        return min(res1, res2, res3, res4)

    space = {
        'max_iter': hp.quniform('max_iter', 100, 500, 1),
        'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag']),
        'C': hp.uniform('C', 0.5, 4),
    }
    best = fmin(fn, space, algo=tpe.suggest, max_evals=5000)
    print(best)








if __name__ == '__main__':
    cv = QCV(qm)


    def fn(params):
        params['max_depth'] = int(params['max_depth'])
        model_id = qm.add_by_params(
            DecisionTreeClassifier(
                max_depth=params['max_depth'],
                max_features=float(params['max_features'])
            ),
            'hyperopt des_tree',
            predict_fn='predict_proba'
        )
        res = cv.cross_val(model_id, 2)
        print(datetime.datetime.now(), res, model_id, params)
        return res


    space = {
        'max_depth': hp.uniform('max_depth', 1, 10),
        'max_features': hp.uniform('max_features', 0.2, 1)
    }
    best = fmin(fn, space, algo=tpe.suggest, max_evals=500)
    print(best)


if __name__ == '__main__':
    cv = QCV(qm)


    def fn(params):
        model_id = qm.add_by_params(
            RandomForestClassifier(
                max_depth=int(params['max_depth']),
                n_estimators=int(params['n_estimators']),
                max_features=float(params['max_features']),
                n_jobs=2
            ),
            'hyperopt rand_forest',
            predict_fn='predict_proba'
        )
        res = cv.cross_val(model_id, 3  )
        print(datetime.datetime.now(), res, model_id, params)
        return res


    space = {
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
        'max_features': hp.uniform('max_features', 0.2, 1)
    }
    best = fmin(fn, space, algo=tpe.suggest, max_evals=5000)
    print(best)



# cv = QCV(qm)
#
# def to1(X, res):
#     ind1 = X[X['numberOfDaysActuallyPlayed'] >= 13].index.values
#     for i in ind1:
#         res.set_value(i, 'res', 1)
#     # ind1 = res[res['res']>0.99].index.values
#     # for i in ind1:
#     #     res.set_value(i, 'res', 1)
#     #
#     # ind1 = res[res['res']<0.02].index.values
#     # for i in ind1:
#     #     res.set_value(i, 'res', 0)
#
#     return list(res['res'])
#
#
# qm.add(
#     35,
#     QPostProcessingModel(
#         2014, 2, to1
#     ),
#     'days13 to res 1'
# )
#
# print(cv.cross_val(2014, 2))
# print(cv.cross_val(35, 2, force=True))
#
#
# #qm.qpredict(33, 2, force=True)


###############

_, conn = get_engine()
cv = QCV(qm)


conn.execute("update qml_models set level=2 where cls in ('qavg', 'qrankedavg')")
res = conn.execute(
    """
        select data_id, cls, descr, 
        substring_index(group_concat(model_id order by cv_score), ',', 50) as models
        from qml_results r 
        inner join qml_models m using(model_id) 
        where m.level=1 and cv_score < 0.389
        group by data_id, cls, descr
    """
).fetchall()

results = []

for r in res:
    for m in r['models'].split(','):
        results.append([int(m), r['data_id']])


for i in range(2000):
    random.shuffle(results)
    models = list(results[:random.randint(2, 7)])
    models = sorted(models, key=lambda x: (x[0], x[1]))
    print(models)

    model_id = qm.add_by_params(
        QAvg(models)
    )
    print(cv.cross_val(model_id, 1))
    model_id = qm.add_by_params(
        QAvg(models, is_geom=True)
    )
    print(cv.cross_val(model_id, 1))
    model_id = qm.add_by_params(
        QRankedAvg(models)
    )
    print(cv.cross_val(model_id, 1))


#
#