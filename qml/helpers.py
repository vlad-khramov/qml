import types
import pickle
from pathlib import Path
from sqlalchemy import create_engine
from qml.config import *


def save(obj, file):
    if isinstance(obj, types.GeneratorType):
        obj = list(obj)

    output = open(file, 'wb')
    pickle.dump(obj, output, protocol=3)
    output.close()


def load(file):
    if not Path(file).is_file():
        return None
    input = open(file, 'rb')
    res = pickle.load(input)
    input.close()
    return res


def get_engine():
    engine = create_engine(QML_DB_CONN_STRING, echo=False)
    conn = engine.connect()
    return engine, conn

e, c = get_engine()
