"""
Plumbing for enabling BPCH file I/O through xarray.
"""
from __future__ import print_function, division

import functools
import os
import numpy as np
import xarray as xr
import warnings

from xarray.core.pycompat import OrderedDict
from xarray.core.variable import  Variable
from xarray.backends.common import AbstractDataStore
from xarray.core.utils import Frozen, FrozenOrderedDict, NDArrayMixin
from xarray.core import indexing

from .. import grid
from .. diagnostics import CTMDiagnosticInfo
from .. io import bpch

from .. tools import ctm2cf, gridspec

DEFAULT_DTYPE = np.dtype('f4')

#: Hard-coded dimension variables to use with any Dataset read in
BASE_DIMENSIONS =   OrderedDict(
    lon=dict(
        dims=['lon', ],
        attrs={
            'standard_name': 'longitude',
            'axis': 'X',
        }
    ),
    lat=dict(
        dims=['lat', ],
        attrs={
            'standard_name': 'latitude',
            'axis': 'Y',
        },
    ),
    time=dict(dims=['time', ], attrs={}),
    nv=dict(dims=['nv', ], attrs={}),
)


#: CF/COARDS recommended dimension order; non-spatiotemporal dimensions
#: should precede these.
DIM_ORDER_PRIORITY = ['time', 'lev', 'lat', 'lon']

class BPCHMixin(object):
    """ Mixin for adding pre-set grid and other information into a BPCH
    dataset. """

    def set_horizontal_grid(self):
        pass

    def _infer_dims(self):
        pass

class BPCHVariableWrapper(indexing.NumpyIndexingAdapter):

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

    @property
    def array(self):
        return self.datastore.ds.variables[self.variable_name].data

    @property
    def dtype(self):
        return np.dtype(self.array.dtype.kind + str(self.array.dtype.itemsize))

    def __getitem__(self, key):
        data = super(BPCHVariableWrapper, self).__getitem__(key)
        copy = self.datastore.memmap
        data = np.array(data, dtype=self.dtype, copy=copy)
        return data


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
        return self.array.dtype

    def __getitem__(self, key):
        if key == () and self.ndim == 0:
            return self.array.get_value()
        return self.array[key]


def open_bpchdataset(filename, fields=[], categories=[],
                     fix_cf=True, fix_dims=False, maskandscale=True,
                     tracerinfo_file='tracerinfo.dat',
                     diaginfo_file='diaginfo.dat',
                     endian=">", default_dtype=DEFAULT_DTYPE,
                     memmap=True, use_dask=True):
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
    categories : list, optional
        List a subset of variable categories to look through. This can
        substantially improve read performance.
    fix_cf : logical, optional
        Conform units, standard names, and other metadata to CF standards when
        reading in data.
    fix_dims : logical, optional
        Transpose dimensions on disk to (T, Z, Y, X) (CF-compliant) order when
        reading in data.
    maskandscale : logical, optional
        Apply scaling to data according to metadata in BPCH file; this might
        slow down some read operations, depending on how eagerly xarray
        tries to evaluate operations.
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
        filename, fields=fields, categories=categories,
        fix_cf=fix_cf,
        tracerinfo_file=tracerinfo_file,
        diaginfo_file=diaginfo_file, endian=endian, memmap=memmap
    )

    ds = xr.Dataset.load_store(store)

    # To immediately load the data from the BPCHDataProxy paylods, need
    # to execute ds.data_vars for some reason...

    return ds

class _BPCHDataStore(AbstractDataStore):
    """Store for reading data via pygchem.io.bpch_file.

    """
    def __init__(self, filename, fields=[], categories=[],
                 mode='r', endian='>', memmap=True,
                 diaginfo_file='', tracerinfo_file='',
                 maskandscale=False, fix_cf=False):

        if not tracerinfo_file:
            tracerinfo_file = 'tracerinfo.dat'
            if not os.path.exists(tracerinfo_file):
                tracerinfo_file = ''
        if not diaginfo_file:
            diaginfo_file = 'diaginfo.dat'
            if not os.path.exists(diaginfo_file):
                diaginfo_file = ''


        # Check endianness flag
        if endian not in ['>', '<', '=']:
            raise ValueError("Invalid byte order (endian={})".format(endian))
        pass
        self.endian = endian


        opener = functools.partial(
            bpch.bpch_file, filename=filename,
            mode=mode, endian=self.endian, memmap=memmap,
            diaginfo_file=diaginfo_file, tracerinfo_file=tracerinfo_file,
            maskandscale=maskandscale)

        self.ds = opener()
        self._opener = opener
        self.memmap = memmap

        self.maskandscale = maskandscale
        self.fields = fields
        self.categories = categories

        self.ctm_info = self.ds.ctm_info

        # Create storage dicts for variables and attributes, to be used later
        # when xarray needs to access the data
        self._variables = xr.core.pycompat.OrderedDict()
        self._attributes = xr.core.pycompat.OrderedDict()
        self._attributes.update(self.ds._attributes)
        self._dimensions = [d for d in BASE_DIMENSIONS]

        # Get the list of variables in the file and load all the data:
        dim_coords = {}
        self._times = self.ds.times
        print(len(self._times))
        self._time_bnds = self.ds.time_bnds
        print(self._attributes)
        self.ctm_grid = grid.CTMGrid.from_model(
            self._attributes['modelname'], resolution=self._attributes['res']
        )

        # Add vertical dimensions
        self._dimensions.append(
            dict(dims=['lev', ], attrs={'axis': 'Z'})
        )
        self._dimensions.append(
            dict(dims=['lev_trop', ], attrs={'axis': 'Z'})
        )
        self._dimensions.append(
            dict(dims=['lev_edge', ], attrs={'axis': 'Z'})
        )

        for vname, v in self.ds.variables.items():
            if fields and (v.attributes['name'] not in fields):
                continue
            if categories and (v.attributes['category ']not in categories):
                continue

            # Process dimensions
            dims = ['time', 'lon', 'lat', ]
            dshape = v.shape
            if len(dshape) > 3:
                # Process the vertical coordinate. A few things can happen here:
                # 1) We have cell-centered values on the "Nlayer" grid; we can take these variables and map them to 'lev'
                # 2) We have edge value on an "Nlayer" + 1 grid; we can take these and use them with 'lev_edge'
                # 3) We have troposphere values on "Ntrop"; we can take these and use them with 'lev_trop', but we won't have coordinate information yet
                # All other cases we do not handle yet; this includes the aircraft emissions and a few other things. Note that tracer sources do not have a vertical coord to worry about!
                nlev = dshape[-1]
                try:
                    if nlev == self.ctm_grid.Nlayers:
                        dims.append('lev')
                    elif nlev == self.ctm_grid.Nlayers + 1:
                        dims.append('lev_edge')
                    elif nlev == self.ctm_grid.Ntrop:
                        dims.append('lev_trop')
                    else:
                        continue
                except AttributeError:
                    warnings.warn("Couldn't resolve attributes on ctm_grid")
                    continue

            # Is the variable time-invariant? If it is, kill the time dim.
            # Here, we mean it only as one sample in the dataset.
            if dshape[0] == 1:
                del dims[0]


            # If requested, try to coerce the attributes and metadata to
            # something a bit more CF-friendly
            lookup_name = vname
            if fix_cf:
                if 'units' in v.attributes:
                    cf_units = ctm2cf.get_cfcompliant_units(
                        v.attributes['units']
                    )
                    v.attributes['units'] = cf_units
                vname = ctm2cf.get_valid_varname(vname)

            # TODO: Explore using a wrapper with an NDArrayMixin; if you don't do this, then dask won't work correctly (it won't promote the data to an array from a BPCHDataProxy). I'm not sure why.
            data = BPCHVariableWrapper(lookup_name, self)
            var = xr.Variable(dims, data, v.attributes)

            # Shuffle dims for CF/COARDS compliance if requested
            # TODO: For this to work, we have to force a load of the data.
            #       Is there a way to re-write BPCHDataProxy so that that's not
            #       necessary?
            #       Actually, we can't even force a load becase var.data is a
            #       numpy.ndarray. Weird.
            # if fix_dims:
            #     target_dims = [d for d in DIM_ORDER_PRIORITY if d in dims]
            #     var = var.transpose(*target_dims)

            self._variables[vname] = var

        # Create the dimension variables; we have a lot of options
        # here with regards to the vertical coordinate. For now,
        # we'll just use the sigma or eta coordinates.
        # Useful CF info: http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#_atmosphere_hybrid_sigma_pressure_coordinate
        # self._variables['Ap'] =
        # self._variables['Bp'] =
        # self._variables['altitude'] =
        if self.ctm_grid.eta_centers is not None:
            lev_vals = self.ctm_grid.eta_centers
            lev_attrs = {
                'standard_name': 'atmosphere_hybrid_sigma_pressure_coordinate',
                'axis': 'Z'
            }
        else:
            lev_vals = self.ctm_grid.sigma_centers
            lev_attrs = {
                'standard_name': 'atmosphere_hybrid_sigma_pressure_coordinate',
                'axis': 'Z'
            }
        self._variables['lev'] = xr.Variable(['lev', ], lev_vals, lev_attrs)

        # Latitude / Longitude
        # TODO: Add lon/lat bounds
        self._variables['lon'] = xr.Variable(
            ['lon'], self.ctm_grid.lonlat_centers[0],
            {'long_name': 'longitude', 'units': 'degrees_east'}
        )
        self._variables['lat'] = xr.Variable(
            ['lat'], self.ctm_grid.lonlat_centers[1],
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
        self._variables['nv'] = xr.Variable(['nv', ], [0, 1])

    def get_variables(self):
        return self._variables

    def get_attrs(self):
        return Frozen(self._attributes)

    def get_dimensions(self):
        return Frozen(self._dimensions)


    def close(self):
        self.ds.close()

    def __exit__(self, type, value, traceback):
        self.close()


class _BPCHDataStore_old(xr.backends.common.AbstractDataStore):
    """ Backend for representing bpch binary output. """

    def __init__(self, filename, fields=[], categories=[],
                 fix_cf=True, fix_dims=False, maskandscale=True,
                 tracerinfo_file='', diaginfo_file='',
                 endian=">", default_dtype=DEFAULT_DTYPE,
                 memmap=True, use_dask=True):
        """
        Note that this should not be used to read in a dataset; instead, see
        open_bpchdataset.

        """

        self._datablocks = None

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
        self._dimensions = [d for d in BASE_DIMENSIONS]

        # Read the BPCH file to figure out what's in it
        read_bpch_kws = dict(
            endian=endian, mode='rb',
            diaginfo_file=diaginfo_file, tracerinfo_file=tracerinfo_file,
            dummy_prefix_dims=1, concat_blocks=True, maskandscale=maskandscale,
            memmap=memmap, use_dask=use_dask
        )
        header_info = bpch.read_bpch(
            filename, first_header=True, **read_bpch_kws
        )
        self.modelname = header_info['modelname']
        self.resolution = header_info['resolution']
        filetype, filetitle, datablocks = bpch.read_bpch(
            filename, **read_bpch_kws
        )
        self.filetype = filetype
        self.filetitle = filetitle
        self._datablocks = datablocks
        if self._datablocks is None:
            raise ValueError("Failed to read the input file " + filename)

        # Get the list of variables in the file and load all the data:
        dim_coords = {}
        self._times = []
        self._time_bnds = []
        self.ctm_grid = grid.CTMGrid.from_model(
            self.modelname, resolution=self.resolution
        )

        # Add vertical dimensions
        self._dimensions.append(
            dict(dims=['lev', ], attrs={'axis': 'Z'})
        )
        self._dimensions.append(
            dict(dims=['lev_trop', ], attrs={'axis': 'Z'})
        )
        self._dimensions.append(
            dict(dims=['lev_edge', ], attrs={'axis': 'Z'})
        )

        for vname, data, dims, attrs in self.load_from_datablocks(datablocks, fields, categories):

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
        if self.ctm_grid.eta_centers is not None:
            lev_vals = self.ctm_grid.eta_centers
            lev_attrs = {
                'standard_name': 'atmosphere_hybrid_sigma_pressure_coordinate',
                'axis': 'Z'
            }
        else:
            lev_vals = self.ctm_grid.sigma_centers
            lev_attrs = {
                'standard_name': 'atmosphere_hybrid_sigma_pressure_coordinate',
                'axis': 'Z'
            }
        self._variables['lev'] = xr.Variable(['lev', ], lev_vals, lev_attrs)

        # Latitude / Longitude
        # TODO: Add lon/lat bounds
        self._variables['lon'] = xr.Variable(
            ['lon'], self.ctm_grid.lonlat_centers[0],
            {'long_name': 'longitude', 'units': 'degrees_east'}
        )
        self._variables['lat'] = xr.Variable(
            ['lat'], self.ctm_grid.lonlat_centers[1],
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
        self._variables['nv'] = xr.Variable(['nv', ], [0, 1])


    def load_from_datablocks(self, datablocks, fields=[], categories=[]):
        """ Process datablocks returned from read_bpch method for use
        in constructing xarray objects. """

        for datablock in datablocks:
            name = ctm2cf.get_valid_varname(datablock['name'])
            category = ctm2cf.get_valid_varname(datablock['category'])

            if fields and (name not in fields): continue
            if category and (category not in categories): continue
            vname =  category + "_" + name
            # if not vname.endswith('O3'): continue

            # Record times for constructing coordinates later
            db_times = datablock['times']
            if not self._times:
                self._time_bnds = db_times
                self._times = [tb[0] for tb in db_times]

            # Increment the shape check here since we're adding a dummy 'time'
            # dimension at the front
            dims = ['time', 'lon', 'lat', ]
            dshape = datablock['data'].shape
            if len(dshape) > 3:
                # Process the vertical coordinate. A few things can happen here:
                # 1) We have cell-centered values on the "Nlayer" grid; we can take these variables and map them to 'lev'
                # 2) We have edge value on an "Nlayer" + 1 grid; we can take these and use them with 'lev_edge'
                # 3) We have troposphere values on "Ntrop"; we can take these and use them with 'lev_trop', but we won't have coordinate information yet
                # All other cases we do not handle yet; this includes the aircraft emissions and a few other things. Note that tracer sources do not have a vertical coord to worry about!
                nlev = dshape[-1]
                try:
                    if nlev == self.ctm_grid.Nlayers:
                        dims.append('lev')
                    elif nlev == self.ctm_grid.Nlayers + 1:
                        dims.append('lev_edge')
                    elif nlev == self.ctm_grid.Ntrop:
                        dims.append('lev_trop')
                    else:
                        continue
                except AttributeError:
                    warnings.warn("Couldn't resolve attributes on ctm_grid")
                    continue

            # Is the variable time-invariant? If it is, kill the time dim.
            # Here, we mean it only as one sample in the dataset.
            if dshape[0] == 1:
                del dims[0]

            # Create a new variable if it's not already present
            tracerinfo = datablock['tracerinfo']
            attrs = dict(
                long_name=tracerinfo['full_name'],
                moulecular_weight=tracerinfo['molecular_weight'],
                scale=tracerinfo['scale'],
                units=tracerinfo['unit']
            )
            data = datablock['data']

            print(vname, dims, data.shape)

            yield vname, data, dims, attrs


    def get_variables(self):
        return self._variables

    def get_attrs(self):
        return Frozen(self._attributes)

    def get_dimensions(self):
        return Frozen(self._dimensions)

    # def close(self):
    #     for var in list(self._variables):
    #         del self._variables['var']
