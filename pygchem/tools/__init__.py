# -*- coding: utf-8 -*-

# parts of pygchem (Python interface for GEOS-Chem Chemistry Transport Model)
#
# Copyright (C) 2013-2014 Benoit Bovy
# see license.txt for more details
#

"""
Utility functions and classes that are used by other modules of PyGChem and
which may present some interest to the user.

This acts as a toolbox gathering miscellaneous functions organized in several
modules depending on their usage or their external dependencies.

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import *
__all__ = ['atm', 'ctm2cf', 'gridspec', 'irisutil', 'timeutil']
