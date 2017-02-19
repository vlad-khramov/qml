import qml.config

qml.config.QML_DATA_DIR = 'workdir/data/'
qml.config.QML_DB_CONN_STRING = 'mysql://root:rootpassword@localhost/temp' #'sqlite:///allstate/foo.db' #

qml.config.QML_TRAIN_X_FILE_MASK = 'workdir/data/train_x_v{0:0=3d}.csv'
qml.config.QML_TEST_X_FILE_MASK = 'workdir/data/test_x_v{0:0=3d}.csv'
qml.config.QML_TRAIN_Y_FILE = 'workdir/data/train_y_v{0:0=3d}.csv'

qml.config.QML_RES_FIELD = 'res'
qml.config.QML_INDEX_COL = 'id'
