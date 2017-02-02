# -*- coding: utf-8 -*-

# parts of pygchem (Python interface for GEOS-Chem Chemistry Transport Model)
#
# Copyright (C) 2012-2014 Gerrit Kuhlmann, Benoît Bovy
# see license.txt for more details
#

"""
Read / write binary punch (BPCH) files.

"""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import *
from past.utils import old_div
from builtins import object
import os

import numpy as np
from dask import delayed
import dask.array as dsa

from pygchem.diagnostics import CTMDiagnosticInfo
from pygchem.tools import timeutil
from pygchem.utils import uff, exceptions


FILETYPE02 = "CTM bin 02"
FILETYPE4D = "CTM bin 4D"
DEFAULT_TITLE = "GEOS-CHEM binary punch file v. 2.0"
ND49_TITLE = "GEOS-CHEM DIAG49 instantaneous timeseries"


class BPCHDataProxy(object):
    """A reference to the data payload of a single BPCH file datablock."""

    __slots__ = ('_shape', 'dtype', 'path', 'endian', 'file_positions',
                 'concat_axis', 'scale_factor', 'fill_value', 'maskandscale', 'memmap', '_data')

    # TODO: Re-factor assuming "concat_axis" is just a prepend option
    def __init__(self, shape, dtype, path, endian, file_positions,
                 concat_axis, scale_factor, fill_value, maskandscale,
                 memmap=True):
        self._shape = shape
        self.dtype = dtype
        self.path = path
        self.fill_value = fill_value
        self.endian = endian
        self.file_positions = file_positions
        self.concat_axis = concat_axis
        self.scale_factor = scale_factor
        self.maskandscale = maskandscale
        self.memmap = memmap
        self._data = None

    @property
    def shape(self):
        if self.concat_axis is None:
            return self._shape
        else:
            shape_bits = list(self._shape)
            shape_bits[self.concat_axis] = len(self.file_positions)
            return tuple(shape_bits)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def data(self):
        if self._data is None:
            if self.memmap:
                self._data = self.load_memmap()
            else:
                self._data = self.load()
        return self._data

    def array(self):
        return self.data

    def load(self):
        with uff.FortranFile(self.path, 'rb', self.endian) as bpch_file:
            all_data = []
            for pos in self.file_positions:
                bpch_file.seek(pos)
                data = np.array(bpch_file.readline('*f'))
                data = data.reshape(self._shape, order='F')
                all_data.append(data)
            data = np.concatenate(all_data, axis=self.concat_axis)
        return data# * self.scale_factor

    def load_memmap(self):
        """ Create a memory-map into the file proxied by this instance.

        note: we offset the file position by 4 bytes to account for the
              prefix that that the Fortran write statement prepends to an
              output line.
        """
        # print("LOAD_MEMMAP", self.shape)
        # all_data = np.empty(self.shape)
        all_data = []
        for i, pos in enumerate(self.file_positions):
            data = delayed(np.memmap)(
                self.path, mode='r',
                dtype=np.dtype(self.endian+'f4'),
                offset=pos+4, shape=self._shape, order='F'
            )
            all_data.append(dsa.from_delayed(data, self._shape, self.dtype))
        all_data = dsa.concatenate(all_data)
        return all_data# * self.scale_factor


    def __getitem__(self, keys):

        if not self.maskandscale:
            return self.data[keys]

        data = self.data[keys].copy()
        if self.scale_factor is not None:
            data = data * self.scale_factor

        return data



    def __repr__(self):
        fmt = '<{self.__class__.__name__} shape={self.shape}' \
              ' dtype={self.dtype!r} path={self.path!r}' \
              ' file_positions={self.file_positions}' \
              ' scale_factor={self.scale_factor}>'
        return fmt.format(self=self)

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def __setstate__(self, state):
        for key, value in list(state.items()):
            setattr(self, key, value)


def read_bpch(filename, mode='rb', skip_data=True,
              diaginfo_file='', tracerinfo_file='',
              dummy_prefix_dims=0, concat_blocks=False, first_header=False,
              maskandscale=True,
              **kwargs):
    """
    Read the binary punch file v2 format.

    Parameters
    ----------
    filename : string
        name or path to the bpch file.
    mode : {'r', 'r+', rb', 'r+b', 'a'}
        file open mode (see :func:`open`). Writing only ('w' or 'wb') is not
        allowed.
    skip_data : bool
        if True, only data block metadata will be read (it will not load data
        into memory but data position and size information will be provided
        in addition to metadata, so that data can further be easily loaded).
    diaginfo_file : string
        path to the 'diaginfo.dat' file (optional). If empty, it will look
        for 'diaginfo.dat' in the same directory than `filename` or it will
        take a default one.
    tracerinfo_file : string
        path to the 'tracerinfo.dat' file (or empty string).
    dummy_prefix_dims : int
        if > 0, then add the indicated number of 'dummy', length-1 dimensions
        to the beginning of each datablock's shape.
    concat_blocks : bool
        if True, logically concatenate blocks from different timestamps during
        read from disk.
    first_header : bool
        if True, only return the header info from the first datablock.
    maskandscale: bool
        if True, return scaled values from the BPCH file
    **kwargs
        extra parameters passed to :class:`pygchem.utils.uff.FortranFile`
        (e.g., `endian`).

    Returns
    -------
    filetype
        bpch file type identifier (given in the file's header)
    filetitle
        title (given in the file's header)
    datablocks
        the list of data blocks, i.e., dictionaries with data block metadata
        and data as a :class:`numpy.ndarray` object or a :class:`BPCHDataProxy`
        instance if `skip_data` is True.

    """
    if mode != 'a' and not mode.endswith('b'):
        mode += 'b'      # platform independent
    if 'w' in mode:
        raise ValueError("write-only mode is not allowed for reading the "
                         "bpch file")

    dir_path = os.path.abspath(os.path.dirname(filename))
    if not dir_path:
        dir_path = os.getcwd()
    if not tracerinfo_file:
        tracerinfo_file = os.path.join(dir_path, "tracerinfo.dat")
        if not os.path.exists(tracerinfo_file):
            tracerinfo_file = ''
    if not diaginfo_file:
        diaginfo_file = os.path.join(dir_path, "diaginfo.dat")
        if not os.path.exists(diaginfo_file):
            diaginfo_file = ''
    ctm_info = CTMDiagnosticInfo(diaginfo_file=diaginfo_file,
                                 tracerinfo_file=tracerinfo_file)

    _PARSED_DATABLOCKS = {}

    with uff.FortranFile(filename, mode, **kwargs) as bpch_file:
        datablocks = []
        filetype = bpch_file.readline().strip()
        fsize = os.path.getsize(filename)
        filetitle = bpch_file.readline().strip()

        while bpch_file.tell() < fsize:
            # read first and second header line
            line = bpch_file.readline('20sffii')
            modelname, res0, res1, halfpolar, center180 = line

            if first_header:
                return dict(
                    modelname=str(modelname, 'utf-8'),
                    resolution=(res0, res1),
                    halfpolar=halfpolar,
                    center180=center180,
                )

            line = bpch_file.readline('40si40sdd40s7i')
            category_name, number, unit, tau0, tau1, reserved = line[:6]
            dim0, dim1, dim2, dim3, dim4, dim5, skip = line[6:]

            # Decode byte-strings to utf-8
            category_name = str(category_name, 'utf-8')
            unit = str(unit, 'utf-8')

            # get additional metadata from tracerinfo / diaginfo
            try:
                cat = ctm_info.categories.select_item(category_name.strip())
                cat_attr = cat.to_dict()
                diag = ctm_info.diagnostics.select_item(
                    cat.offset + int(number)
                )
                diag_attr = diag.to_dict()
                if not unit.strip():            # unit may be empty in bpch
                    unit = diag_attr['unit']    # but not in tracerinfo
            except exceptions.SelectionMismatchError:
                diag = {'name': '', 'scale': 1}
                diag_attr = {}
                cat_attr = {}

            vname = diag['name']
            fullname = category_name.strip() + "_" + vname

            # parse metadata, get data or set a data proxy
            concat_axis = None
            if dim2 == 1:
                data_shape = (dim0, dim1)         # 2D field
            else:
                data_shape = (dim0, dim1, dim2)
            if dummy_prefix_dims > 0:
                concat_axis = 0
                data_shape = tuple([1, ]*dummy_prefix_dims + list(data_shape))

            from_file = os.path.abspath(filename)
            file_position = bpch_file.tell()
            if skip_data:
                bpch_file.skipline()
                if concat_blocks and (fullname in _PARSED_DATABLOCKS):
                    data = _PARSED_DATABLOCKS[fullname]['data']
                    data.file_positions.append(file_position)
                    times = _PARSED_DATABLOCKS[fullname]['times']
                    times.append(
                        (timeutil.tau2time(tau0), timeutil.tau2time(tau1))
                    )
                    continue
                else:
                    data = BPCHDataProxy(data_shape, np.dtype('f'),
                                         from_file, bpch_file.endian,
                                         [file_position, ], concat_axis, diag['scale'], np.nan, maskandscale)
            else:
                # TODO: Converting the BPCHDataProxy to record multiple file
                #       positions breaks the symmetry with load-on-read below,
                #       so we should ideally perform a similar sort of concatenation
                #       operation here, including recording multiple timestamps.
                data = np.array(bpch_file.readline('*f'))
                data = data.reshape((dim0, dim1, dim2), order='F')

            datablock = {'number': int(number),
                         'name': vname,
                         'category': category_name.strip(),
                         'times': [(timeutil.tau2time(tau0),
                                    timeutil.tau2time(tau1)), ],
                         'modelname': modelname.strip(),
                         'center180': bool(center180),
                         'halfpolar': bool(halfpolar),
                         'origin': (dim3, dim4, dim5),
                         'resolution': (res0, res1),
                         'shape': data_shape,
                         'loaded_from_file': from_file,
                         'save_to_file': '',
                         'file_position': file_position,
                         'data': data,
                         'unit': unit.strip(),
                         'tracerinfo': diag_attr,
                         'diaginfo': cat_attr}
            datablocks.append(datablock)
            _PARSED_DATABLOCKS[fullname] = datablock

    return filetype, filetitle, datablocks


def create_bpch(filename, title=DEFAULT_TITLE, filetype=FILETYPE02, **kwargs):
    """
    Create a new empty bpch file.

    Parameters
    ----------
    filename : string
        name or path to the bpch file.
    title : string
        a title line to write in the file's header.
    filetype : string
        bpch file type identifier (either :attr:`bpch.FILETYPE02` or
        :attr:`bpch.FILETYPE4D`).
    **kwargs
        extra parameters passed to :class:`pygchem.utils.uff.FortranFile`
        (e.g., `endian`).

    Returns
    -------
    bpch_file
        the open file instance (:class:`pygchem.utils.uff.FortranFile` object).

    """
    bpch_file = uff.FortranFile(filename, 'wb', **kwargs)
    bpch_file.writeline('40s', filetype.ljust(40))
    bpch_file.writeline('80s', title.ljust(80))
    return bpch_file


def append_bpch(bpch_file, datablock):
    """
    Append a data block to an open bpch file.

    Parameters
    ----------
    bpch_file : file object
        bpch file (with writing permissions), as returned by
        :func:`read_bpch` or :func:`create_bpch`.
    datablock : dict
        data block metadata and data.

    """
    if isinstance(datablock['data'], BPCHDataProxy):
        data = old_div(datablock['data'].data, datablock['data'].scale_factor)
    else:
        data = datablock['data']

    if len(datablock['shape']) == 2:
        data_shape = datablock['shape'] + (1,)    # 2D field
    else:
        data_shape = datablock['shape']

    bpch_file.writeline(
        '20sffii',
        datablock['modelname'].ljust(20),
        datablock['resolution'][0], datablock['resolution'][1],
        datablock['halfpolar'], datablock['center180']
    )
    bpch_file.writeline(
        '40si40s2d40s7i',
        datablock['category'].ljust(40),
        datablock['number'], datablock['unit'].ljust(40),
        timeutil.time2tau(datablock['times'][0]),
        timeutil.time2tau(datablock['times'][1]),
        ''.ljust(40),
        data_shape[0], data_shape[1], data_shape[2],
        datablock['origin'][0], datablock['origin'][1], datablock['origin'][2],
        data.size * 4
    )
    data_array = data.flatten('F')
    bpch_file.writeline('%df' % data.size, *data_array)


def write_bpch(filename, datablocks, title=DEFAULT_TITLE,
               filetype=FILETYPE02, **kwargs):
    """
    Write data blocks to the binary punch file v2 format.

    Parameters
    ----------
    filename : string
        name or path to the bpch file.
    datablocks : sequence of dicts
        data blocks metadata and data.
    title : string
        a title line to write in the file's header.
    filetype : string
        bpch file type identifier (either :attr:`bpch.FILETYPE02` or
        :attr:`bpch.FILETYPE4D`).
    **kwargs
        extra parameters passed to :class:`pygchem.utils.uff.FortranFile`
        (e.g., `endian`).

    """
    with create_bpch(filename, title, filetype, **kwargs) as bpch_file:
        for db in datablocks:
            append_bpch(bpch_file, db)
