import qml.config as config

config.QML_DATA_DIR = 'workdir/data/'
config.QML_DB_CONN_STRING = 'mysql://root:rootpassword@localhost/temp' #'sqlite:///allstate/foo.db' #

config.QML_TRAIN_X_FILE_MASK = 'workdir/data/v{0:0=4d}_train_x.csv'
config.QML_TEST_X_FILE_MASK = 'workdir/data/v{0:0=4d}_test_x.csv'
config.QML_TRAIN_Y_FILE_MASK = 'workdir/data/v{0:0=4d}_train_y.csv'

config.QML_RES_COL   = 'res'
config.QML_INDEX_COL = 'id'
