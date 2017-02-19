import workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb
from workdir.classes.models import qm


cv = QCV(qm)

cv.cross_val(1, 1)