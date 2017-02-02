"""
Plumbing for enabling BPCH file I/O through xarray.
"""
from __future__ import print_function, division

import os
import numpy as np
import xarray as xr

from xarray.core.pycompat import OrderedDict
from xarray.core.utils import Frozen, FrozenOrderedDict, NDArrayMixin
from xarray.core import indexing

from .. import grid
from .. diagnostics import CTMDiagnosticInfo
from .. io import bpch

from .. tools import ctm2cf

DEFAULT_DTYPE = np.dtype('f4')

#: Hard-coded dimension variables to use with any Dataset read in
DIMENSIONS = OrderedDict(
    lon=dict(dims=['lon', ],
             attrs={
                'standard_name': 'longitude',
                'axis': 'X',
            }
    ),
    lat=dict(dims=['lat', ],
             attrs={
                'standard_name': 'latitude',
                'axis': 'Y',
             },
    ),
    lev=dict(dims=['lev', ],
             attrs={
                'axis': 'Z',
             }
    ),
    time=dict(dims=['time', ],
              attrs={}

    ),
    nv=dict(),
)

#: CF/COARDS recommended dimension order; non-spatiotemporal dimensions
#: should precede these.
DIM_ORDER_PRIORITY = ['time', 'lev', 'lat', 'lon']


class BPCHDataProxyWrapper(NDArrayMixin):
    # Mostly following the template of ScipyArrayWrapper
    # https://github.com/pydata/xarray/blob/master/xarray/backends/scipy_.py
    # and NioArrayWrapper
    # https://github.com/pydata/xarray/blob/master/xarray/backends/pynio_.py

    def __init__(self, array):
        self._array = array

    @property
    def array(self):
        return self._array.data

    @property
    def dtype(self):
        return np.dtype(self.array.dtype.kind + str(self.array.dtype.itemsize))

    # def __getitem__(self, key):
    #     data = super(BPCHDataProxyWrapper, self).__getitem__(key)
    #     data = np.array(data, dtype=self.dtype, copy=True)
    #     return data



def open_bpchdataset(filename, fields=[], fix_cf=True, fix_dims=False,
                     tracerinfo_file='tracerinfo.dat',
                     diaginfo_file='diaginfo.dat',
                     endian=">", default_dtype=DEFAULT_DTYPE, chunks=None):
    """ Open a GEOS-Chem BPCH file output as an xarray Dataset.

    Parameters
    ----------
    filename : string
        Path to the output file to read in.
    {tracerinfo,diaginfo}_file : string, optional
        Path to the metadata "info" .dat files which are used to decipher
        the metadata corresponding to each variable in the output dataset.
        If not provided, will look for them in the current directory or
        fall back on a generic set.
    fields : list, optional
        List of a subset of variable names to return. This can substantially
        improve read performance. Note that the field here is just the tracer
        name - not the category, e.g. 'O3' instead of 'IJ-AVG-$_O3'.
    fix_cf : logical, optional
        Conform units, standard names, and other metadata to CF standards when
        reading in data.
    fix_dims : logical, optional
        Transpose dimensions on disk to (T, Z, Y, X) (CF-compliant) order when
        reading in data.
    endian : {'=', '>', '<'}, optional
        Endianness of file on disk. By default, "big endian" (">") is assumed.
    default_dtype : numpy.dtype, optional
        Default datatype for variables encoded in file on disk (single-precision
        float by default).

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the requested fields (or the entire file), with data
        contained in proxy containers for access later.

    """

    # Do a preliminary read of the BPCH file to construct a map of its
    # contents. This doesn't read anything into memory, and should be
    # very fast
    # filetype, filetitle, datablocks = bpch.read_bpch(
    #     filename, skip_data=True,
    #     diaginfo_file=diaginfo_file, tracerinfo_file=tracerinfo_file
    # )
    # bpch_contents = _get_bpch_contents(datablocks)

    store = _BPCHDataStore(
        filename, fields=fields, fix_cf=fix_cf, fix_dims=fix_dims,
        tracerinfo_file=tracerinfo_file,
        diaginfo_file=diaginfo_file, endian=endian,
        default_dtype=default_dtype,
    )

    ds = xr.Dataset.load_store(store)

    # To immediately load the data from the BPCHDataProxy paylods, need
    # to execute ds.data_vars for some reason...

    return ds


class _BPCHDataStore(xr.backends.common.AbstractDataStore):
    """ Backend for representing bpch binary output. """

    def __init__(self, filename, fields=[], fix_cf=True, fix_dims=False,
                 tracerinfo_file='', diaginfo_file='',
                 endian=">", default_dtype=DEFAULT_DTYPE):
        """
        Note that this should not be used to read in a dataset; instead, see
        open_bpchdataset.

        """

        # Cache the diagnostic info from this simulation for accessing
        # grid and metadata information
        if not tracerinfo_file:
            tracerinfo_file = 'tracerinfo.dat'
            if not os.path.exists(tracerinfo_file):
                tracerinfo_file = ''
        if not diaginfo_file:
            diaginfo_file = 'diaginfo.dat'
            if not os.path.exists(diaginfo_file):
                diaginfo_file = ''

        self.ctm_info = CTMDiagnosticInfo(
            diaginfo_file=diaginfo_file, tracerinfo_file=tracerinfo_file
        )

        # Check endianness flag
        if endian not in ['>', '<', '=']:
            raise ValueError("Invalid byte order (endian={})".format(endian))
        pass
        self.endian = endian

        # Create storage dicts for variables and attributes, to be used later
        # when xarray needs to access the data
        self._variables = xr.core.pycompat.OrderedDict()
        self._attributes = xr.core.pycompat.OrderedDict()
        self._dimensions = [d for d in DIMENSIONS]

        # Read the BPCH file to figure out what's in it
        read_bpch_kws = dict(
            endian=endian, mode='rb',
            diaginfo_file=diaginfo_file, tracerinfo_file=tracerinfo_file,
            dummy_prefix_dims=1, concat_blocks=True
        )
        header_info = bpch.read_bpch(
            filename, first_header=True, **read_bpch_kws
        )
        filetype, filetitle, datablocks = bpch.read_bpch(
            filename, **read_bpch_kws
        )

        # Get the list of variables in the file and load all the data:
        dim_coords = {}
        self._times = []
        self._time_bnds = []
        ctm_grid = grid.CTMGrid.from_model(
            header_info['modelname'], resolution=header_info['resolution']
        )

        for vname, data, dims, attrs in self.load_from_datablocks(datablocks, fields):

            # If requested, try to coerce the attributes and metadata to
            # something a bit more CF-friendly
            if fix_cf:
                if 'units' in attrs:
                    cf_units = ctm2cf.get_cfcompliant_units(attrs['units'])
                    attrs['units'] = cf_units
                vname = ctm2cf.get_valid_varname(vname)

            # data = data.load_memmap()
            # TODO: Explore using a wrapper with an NDArrayMixin; if you don't do this, then dask won't work correctly (it won't promote the data to an array from a BPCHDataProxy). I'm not sure why.
            data = BPCHDataProxyWrapper(data)
            var = xr.Variable(dims, data, attrs)

            # Shuffle dims for CF/COARDS compliance if requested
            # TODO: For this to work, we have to force a load of the data.
            #       Is there a way to re-write BPCHDataProxy so that that's not
            #       necessary?
            #       Actually, we can't even force a load becase var.data is a
            #       numpy.ndarray. Weird.
            if fix_dims:
                target_dims = [d for d in DIM_ORDER_PRIORITY if d in dims]
                var = var.transpose(*target_dims)

            self._variables[vname] = var

        # Create the dimension variables; we have a lot of options
        # here with regards to the vertical coordinate. For now,
        # we'll just use the sigma or eta coordinates.
        # Useful CF info: http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#_atmosphere_hybrid_sigma_pressure_coordinate
        # self._variables['Ap'] =
        # self._variables['Bp'] =
        # self._variables['altitude'] =
        if ctm_grid.eta_centers is not None:
            lev_vals = ctm_grid.eta_centers
            lev_attrs = {
                'standard_name': 'atmosphere_hybrid_sigma_pressure_coordinate',
                'axis': 'Z'
            }
        else:
            lev_vals = ctm_grid.sigma_centers
            lev_attrs = {
                'standard_name': 'atmosphere_sigma_coordinate',
                'axis': 'Z'
            }
        self._variables['lev'] = xr.Variable(['lev', ], lev_vals, lev_attrs)

        # Latitude / Longitude
        # TODO: Add lon/lat bounds
        self._variables['lon'] = xr.Variable(
            ['lon'], ctm_grid.lonlat_centers[0],
            {'long_name': 'longitude', 'units': 'degrees_east'}
        )
        self._variables['lat'] = xr.Variable(
            ['lat'], ctm_grid.lonlat_centers[1],
            {'long_name': 'latitude', 'units': 'degrees_north'}
        )
        # TODO: Fix longitudes if ctm_grid.center180

        # Time dimensions
        # TODO: Time units?
        self._variables['time'] = xr.Variable(
            ['time', ], self._times,
            {'bounds': 'time_bnds', 'units': ctm2cf.CTM_TIME_UNIT_STR}
        )
        self._variables['time_bnds'] = xr.Variable(
            ['time', 'nv'], self._time_bnds,
            {'units': ctm2cf.CTM_TIME_UNIT_STR}
        )

    def load_from_datablocks(self, datablocks, fields=[]):
        """ Process datablocks returned from read_bpch method for use
        in constructing xarray objects. """

        for datablock in datablocks:
            name = datablock['name']
            if fields and (name not in fields): continue
            vname = datablock['category'] + "_" + datablock['name']
            # if not vname.endswith('O3'): continue

            # Record times for constructing coordinates later
            db_times = datablock['times']
            if not self._times:
                self._time_bnds = db_times
                self._times = [tb[0] for tb in db_times]

            # Increment the shape check here since we're adding a dummy 'time'
            # dimension at the front
            if len(datablock['shape']) == 3:
                dims = ['lon', 'lat', ]
            else:
                dims = ['lon', 'lat', 'lev', ]
            dims = ['time', ] + dims

            # Create a new variable if it's not already present
            tracerinfo = datablock['tracerinfo']
            attrs = dict(
                long_name=tracerinfo['full_name'],
                moulecular_weight=tracerinfo['molecular_weight'],
                scale=tracerinfo['scale'],
                units=tracerinfo['unit']
            )
            data = datablock['data']

            yield vname, data, dims, attrs


    def _get_datablock_dims(datablock):
        """
        Compute the grid dimensions for a given datablock.

        This is mostly culled from backend_iris._get_datablock_dim_coords
        and irisutil.coord_from_grid, with some modifications for
        pertinence to xarray.

        """
        modelname = str(datablock['modelname'], 'utf-8')

        ctm_grid = grid.CTMGrid.from_model(
            modelname, resolution=datablock['resolution']
        )

        # verify if `ctm_grid` is compatible with requested coordinates
        if ctm_grid.Nlayers is None and any(c in vcoord_names for c in coord):
            raise ValueError("vertical coordinate(s) requested but the '{0}' grid "
                             "has no defined vertical layers"
                             .format(ctm_grid.model))
        if 'sigma' in coord and ctm_grid.hybrid:
            raise ValueError("sigma coordinate requested but the '{0}' grid have "
                             "hybrid vertical layers ".format(ctm_grid.model))
        if 'eta' in coord and not ctm_grid.hybrid:
            raise ValueError("eta coordinate requested but the '{0}' grid doesn't "
                             "have hybrid vertical layers ".format(ctm_grid.model))

        # Reshape coordinate bounds from shape (n+1) to shape (n, 2).
        reshape_bounds = lambda bounds: np.column_stack((bounds[:-1], bounds[1:]))

        dims = [
            ()
        ]

        return dims

    def get_variables(self):
        return self._variables

    def get_attrs(self):
        return Frozen(self._attributes)

    def get_dimensions(self):
        return Frozen(self._dimensions)

    # def close(self):
    #     for var in list(self._variables):
    #         del self._variables['var']
