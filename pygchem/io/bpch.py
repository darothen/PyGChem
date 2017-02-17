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
from collections import OrderedDict
import mmap as mm
import os
import warnings

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

    # __slots__ = ('_shape', 'dtype', 'path', 'endian', 'file_position',
    #              'scale_factor', 'fill_value', 'maskandscale', '_data')

    def __init__(self, shape, dtype, path, endian, file_position,
                 scale_factor, fill_value, maskandscale):
        self._shape = shape
        self.dtype = dtype
        self.path = path
        self.fill_value = fill_value
        self.endian = endian
        self.file_position = file_position
        self.scale_factor = scale_factor
        self.maskandscale = maskandscale
        self._data = None

    def _maybe_mask_and_scale(self, arr):
        if self.maskandscale and (self.scale_factor is not None):
            arr = self.scale_factor * arr
        return arr

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def data(self):
        if self._data is None:
            self._data = self.load()
        return self._data

    def array(self):
        return self.data

    def load(self):
        with uff.FortranFile(self.path, 'rb', self.endian) as bpch_file:
            pos = self.file_position
            bpch_file.seek(pos)
            data = np.array(bpch_file.readline('*f'))
            data = data.reshape(self._shape, order='F')
        if self.maskandscale and (self.scale_factor is not None):
            data = data * self.scale_factor
        return data

    def __getitem__(self, keys):
        return self.data[keys]

    def __repr__(self):
        fmt = '<{self.__class__.__name__} shape={self.shape}' \
              ' dtype={self.dtype!r} path={self.path!r}' \
              ' file_position={self.file_position}' \
              ' scale_factor={self.scale_factor}>'
        return fmt.format(self=self)

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def __setstate__(self, state):
        for key, value in list(state.items()):
            setattr(self, key, value)

class BPCHDataProxyConcatDim(object):
    """A reference to the data payload of a single BPCH file datablock."""

    __slots__ = ('_shape', 'dtype', 'path', 'endian', 'file_positions',
                 'concat_axis', 'scale_factor', 'fill_value', 'maskandscale',
                 'memmap', 'use_dask', '_data')

    # TODO: Re-factor assuming "concat_axis" is just a prepend option
    def __init__(self, shape, dtype, path, endian, file_positions,
                 concat_axis, scale_factor, fill_value, maskandscale,
                 memmap=True, use_dask=True):
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
        self.use_dask = use_dask
        self._data = None

    @property
    def shape(self):
        if (self.concat_axis is None) or (len(self.file_positions) == 1):
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
                # print("LOAD (memmap)")
                self._data = self.load_memmap()
            else:
                # print("LOAD")
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

            #     if self.use_dask:
            #         print("   DASK")
            #         print(np.prod(self._shape),)
            #         data = delayed(bpch_file.readline)('*f')
            #         data = dsa.from_delayed(
            #             data, shape=(np.prod(self._shape), ), dtype=self.dtype
            #         )
            #         data = dsa.reshape(data, self._shape, order='F')
            #     else:
            #         data = np.array(bpch_file.readline('*f'))
            #         data = data.reshape(self._shape, order='F')
            #     all_data.append(data)
            #
            # if self.use_dask:
            #     data = dsa.concatenate(all_data, axis=self.concat_axis)
            # else:
            #     data = np.concatenate(all_data, axis=self.concat_axis)

        if self.maskandscale and (self.scale_factor is not None):
            data = data * self.scale_factor
        return data


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

            if self.use_dask:
                # print("   DASK")
                data = delayed(np.memmap)(
                    self.path, mode='r',
                    dtype=np.dtype(self.endian+'f4'),
                    offset=pos+4, shape=self._shape, order='F'
                )
                data = dsa.from_delayed(data, self._shape, self.dtype)

            else:
                data = np.memmap(self.path, mode='r',
                                 dtype=np.dtype(self.endian+'f4'),
                                 offset=pos+4, shape=self._shape, order='F')
            all_data.append(data)

        if len(all_data) > 1:
            if self.use_dask:
                data = dsa.concatenate(all_data, axis=self.concat_axis)
            else:
                data = np.concatenate(all_data, axis=self.concat_axis)
        else:
            # In this case, the data is time-invariant and we should prefer
            # to squeeze the leading dimension
            data = all_data[0][0]

        if self.maskandscale and (self.scale_factor is not None):
            data = data * self.scale_factor

        return data


    def __getitem__(self, keys):
        return self.data[keys]

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


class bpch_variable(BPCHDataProxy):

    def __init__(self, data_pieces, *args, attributes=None,
                 **kwargs):
        self._data_pieces = data_pieces
        super(bpch_variable, self).__init__(*args, **kwargs)

        if attributes is not None:
            self._attributes = attributes
        else:
            self._attributes = OrderedDict()
        for k, v in self._attributes.items():
            self.__dict__[k] = v

    def load(self):
        warnings.warn("'load' disabled on bpch_variable.")
        pass

    @property
    def shape(self):
        shape_copy = list(self._shape)
        shape_copy[0] = len(self._data_pieces)
        return tuple(shape_copy)

    @property
    def chunks(self):
        return len(self._data_pieces)

    @property
    def data(self):
        arr = np.concatenate(self._data_pieces, axis=0).view(self.dtype)
        return self._maybe_mask_and_scale(arr)

    @property
    def attributes(self):
        return self._attributes

    def __setattr__(self, key, value):
        try:
            self._attribute[key] = value
        except AttributeError:
            pass
        self.__dict__[key] = value

    def __str__(self):
        pass

    def __repr__(self):
        return """
          <{self.__class__.__name__}
          shape={self.shape} (chunks={self.chunks})
           dtype={self.dtype!r} scale_factor={self.scale_factor}>
        """.format(self=self)

    def __getitem__(self, index):
        if self.chunks == 1:
            arr = self._data_pieces[0][index]
        else:
            pass

        return self._maybe_mask_and_scale(arr)

class bpch_file(object):
    """ A file object for BPCH data on disk.

    Parameters
    ----------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, filename, mode='rb', endian='>', memmap=False,
                 diaginfo_file='', tracerinfo_file='',
                 maskandscale=False):
        """ Initialize a bpch_file. """

        self.mode = mode
        if not mode.startswith('r'):
            raise ValueError("Currently only know how to 'r(b)'ead bpch files.")

        self.filename = filename
        self.fsize = os.path.getsize(self.filename)
        self.use_mmap = memmap
        self.endian = endian

        # Open a pointer to the file
        self.fp = uff.FortranFile(self.filename, self.mode, self.endian)

        # self._mm = None
        # self._mm_buf = None
        # if self.use_mmap:
        #     self._mm = mm.mmap(self.fp.fileno(), 0, access=mm.ACCESS_READ)
        #     self._mm_buf = np.frombuffer(self._mm, dtype=np.int8)

        dir_path = os.path.abspath(os.path.dirname(filename))
        if not dir_path:
            dir_path = os.getcwd()
        if not tracerinfo_file:
            tracerinfo_file = os.path.join(dir_path, "tracerinfo.dat")
            if not os.path.exists(tracerinfo_file):
                tracerinfo_file = ''
        self.tracerinfo_file = tracerinfo_file
        if not diaginfo_file:
            diaginfo_file = os.path.join(dir_path, "diaginfo.dat")
            if not os.path.exists(diaginfo_file):
                diaginfo_file = ''
        self.diaginfo_file = diaginfo_file
        self.ctm_info = CTMDiagnosticInfo(diaginfo_file=self.diaginfo_file,
                                          tracerinfo_file=self.tracerinfo_file)

        # self.dimensions = OrderedDict()
        self.variables = OrderedDict()
        self.times = []
        self.time_bnds = []

        self._attributes = OrderedDict()

        # Critical information for accessing file contents
        self.maskandscale = maskandscale
        self._header_pos = None

        if mode.startswith('r'):
            self._read()


    def close(self):
        """ Close this bpch file. """
        import weakref
        import warnings

        if not self.fp.closed:
            self.variables = OrderedDict()

            # if self._mm_buf is not None:
            #     ref = weakref.ref(self._mm_buf)
            #     self._mm_buf = None
            #     if ref() is None:
            #         self._mm.close()
            #     else:
            #         warnings.warn(
            #             "Can't close bpch_file opened with memory mapping until "
            #             "all of variables/arrays referencing its data are "
            #             "copied and/or cleaned", category=RuntimeWarning)
            # self._mm = None
            self.fp.close()
    # __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _read(self):
        """ Parse the file on disk and set up easy access to meta-
        and data blocks """

        self._read_metadata()
        self._read_header()
        self._read_var_data()


    def _read_metadata(self):
        """ Read the main metadata packaged within a bpch file """

        filetype = self.fp.readline().strip()
        filetitle = self.fp.readline().strip()

        self.__setattr__('filetype', filetype)
        self.__setattr__('filetitle', filetitle)


    def _read_header(self, all_info=False):

        self._header_pos = self.fp.tell()

        line = self.fp.readline('20sffii')
        modelname, res0, res1, halfpolar, center180 = line
        self._attributes.update({
            "modelname":str(modelname, 'utf-8').strip(),
            "halfpolar": halfpolar,
            "center180": center180,
            "res": (res0, res1)
        })
        self.__setattr__('modelname', modelname)
        self.__setattr__('res', (res0, res1))
        self.__setattr__('halfpolar', halfpolar)
        self.__setattr__('center180', center180)

        # Re-wind the file
        self.fp.seek(self._header_pos)


    def _read_var_data(self):

        var_bundles = OrderedDict()
        var_attrs = OrderedDict()
        _times = []

        n_vars = 0

        while self.fp.tell() < self.fsize:

            var_attr = OrderedDict()

            # read first and second header lines
            line = self.fp.readline('20sffii')
            modelname, res0, res1, halfpolar, center180 = line

            line = self.fp.readline('40si40sdd40s7i')
            category_name, number, unit, tau0, tau1, reserved = line[:6]
            dim0, dim1, dim2, dim3, dim4, dim5, skip = line[6:]
            var_attr['number'] = number

            # Decode byte-strings to utf-8
            category_name = str(category_name, 'utf-8')
            var_attr['category'] = category_name.strip()
            unit = str(unit, 'utf-8')

            # get additional metadata from tracerinfo / diaginfo
            try:
                cat = self.ctm_info.categories.select_item(
                    category_name.strip())
                cat_attr = cat.to_dict()
                diag = self.ctm_info.diagnostics.select_item(
                    cat.offset + int(number)
                )
                diag_attr = diag.to_dict()
                if not unit.strip():  # unit may be empty in bpch
                    unit = diag_attr['unit']  # but not in tracerinfo
                var_attr.update(diag_attr)
            except exceptions.SelectionMismatchError:
                diag = {'name': '', 'scale': 1}
                var_attr.update(diag)
                diag_attr = {}
                cat_attr = {}
            var_attr['unit'] = unit

            vname = diag['name']
            fullname = category_name.strip() + "_" + vname
            # print(fullname)

            # parse metadata, get data or set a data proxy
            if dim2 == 1:
                data_shape = (dim0, dim1)         # 2D field
            else:
                data_shape = (dim0, dim1, dim2)
            # Add proxy time dimension to shape
            data_shape = tuple([1, ] + list(data_shape))
            origin = (dim3, dim4, dim5)
            var_attr['origin'] = origin

            pos = self.fp.tell()

            # Map or read the data
            if self.use_mmap:
                dtype = np.dtype(self.endian + 'f4')
                offset = pos + 4
                data = np.memmap(self.filename, mode='r',
                                 shape=data_shape,
                                 dtype=dtype, offset=offset, order='F')
                # print(len(data), data_shape, np.product(data_shape))
                # data.shape = data_shape
                self.fp.skipline()
            else:
                self.fp.seek(pos)
                data = np.array(self.fp.readline('*f'))
                # data = data.reshape(data_shape, order='F')
                data.shape = data_shape
            # Save the data as a "bundle" for concatenating in the final step
            if fullname in var_bundles:
                var_bundles[fullname].append(data)
            else:
                var_bundles[fullname] = [data, ]
                var_attrs[fullname] = var_attr
                n_vars += 1

            timelo, timehi = timeutil.tau2time(tau0), timeutil.tau2time(tau1)
            _times.append((timelo, timehi))

        # Copy over the data we've recorded
        self.time_bnds[:] = _times[::n_vars]
        self.times = [t[0] for t in _times[::n_vars]]

        for fullname, bundle in var_bundles.items():
            var_attr = var_attrs[fullname]
            self.variables[fullname] = bpch_variable(
                bundle, data_shape, np.dtype('f'), self.filename, self.endian,
                file_position=None, scale_factor=var_attr['scale'],
                fill_value=np.nan, maskandscale=False, attributes=var_attr
            )

# TODO: Incremental read mode?
def read_bpch(filename, mode='rb', skip_data=True,
              diaginfo_file='', tracerinfo_file='',
              dummy_prefix_dims=0, concat_blocks=False, first_header=False,
              maskandscale=True, memmap=True, use_dask=True,
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
                    data = BPCHDataProxyConcatDim(
                        data_shape, np.dtype('f'),
                        from_file, bpch_file.endian,
                        [file_position, ], concat_axis, diag['scale'], np.nan, maskandscale,
                        memmap=memmap, use_dask=use_dask
                    )
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


BPCHFile = bpch_file