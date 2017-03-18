from sklearn.neighbors import KNeighborsClassifier

import workdir.classes.config
from qml.models import QModels, QXgb, QAvg

qm = QModels.get_instance()

qm.add(1,
    QXgb(
        booster='gbtree',
        objective='binary:logistic',
        eval_metric='logloss',
        subsample=0.5,
        eta=0.1,#learn_rate
        max_depth=3,
        num_boost_round=100
    ),
    'simple xgb'
)

qm.add(2,
    QXgb(
        booster='gbtree',
        objective='binary:logistic',
        eval_metric='logloss',
        subsample=0.5,
        eta=0.006,#learn_rate
        max_depth=4,
        num_boost_round=3400
    )
)





# qm.add(21, SVC(kernel="linear", C=0.25, probability=True), predict_fn='predict_proba')
# qm.add(22, SVC(gamma=2, C=1, probability=True), predict_fn='predict_proba')
# qm.add(23, GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True), predict_fn='predict_proba', description_params='1.0 * RBF(1.0), warm_start=True')
# qm.add(24, DecisionTreeClassifier(max_depth=5), predict_fn='predict_proba')
# qm.add(25, RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), predict_fn='predict_proba')
# qm.add(26, MLPClassifier(alpha=1), predict_fn='predict_proba')
# qm.add(27, AdaBoostClassifier(), predict_fn='predict_proba')
# qm.add(28, GaussianNB(), predict_fn='predict_proba')
# qm.add(29, QuadraticDiscriminantAnalysis(), predict_fn='predict_proba')
#
