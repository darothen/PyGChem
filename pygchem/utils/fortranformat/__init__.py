from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
__version__ = '0.2.3'

import sys
IS_PYTHON3 = sys.version_info[0] >= 3

if IS_PYTHON3:
    exec('from .FortranRecordReader import FortranRecordReader')
    exec('from .FortranRecordWriter import FortranRecordWriter')
    exec('from . import config')
else:
    exec('from FortranRecordReader import FortranRecordReader')
    exec('from FortranRecordWriter import FortranRecordWriter')
    exec('import config')


