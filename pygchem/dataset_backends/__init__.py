# -*- coding: utf-8 -*-

"""
Backends used to read/write GEOS-Chem datasets
(netCDF and/or bpch formats are supported, depending on the backend).


Each backend should be implemented in a Python module named as follows:

    backend_`name`.py

where `name` is the name of the backend or the python library used as backend.

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import *