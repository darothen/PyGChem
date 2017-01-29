# -*- coding: utf-8 -*-

# parts of pygchem: Python interface for GEOS-Chem Chemistry Transport Model
#
# Copyright (C) 2013-2014 Benoit Bovy
# see license.txt for more details
#

"""
Read/write GEOS-Chem files (input or output) into/from Python built-in
data structures.

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import *
__all__ = ["globchem", "bpch", "diaginfo"]
