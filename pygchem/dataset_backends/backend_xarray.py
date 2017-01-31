"""
Plumbing for enabling BPCH file I/O through xarray.
"""
from __future__ import print_function, division

import os
import numpy as np
import xarray as xr

from xarray.core.pycompat import OrderedDict

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
              attrs={
                'long_name': 'time',
                'bounds': 'time_bnds',
              }
    ),
    nv=dict(),
)

def open_bpchdataset(filename, fields=[], fix_cf=True,
                     tracerinfo_file='tracerinfo.dat',
                     diaginfo_file='diaginfo.dat',
                     endian=">", default_dtype=DEFAULT_DTYPE):

    # Do a preliminary read of the BPCH file to construct a map of its
    # contents. This doesn't read anything into memory, and should be
    # very fast
    # filetype, filetitle, datablocks = bpch.read_bpch(
    #     filename, skip_data=True,
    #     diaginfo_file=diaginfo_file, tracerinfo_file=tracerinfo_file
    # )
    # bpch_contents = _get_bpch_contents(datablocks)

    store = _BPCHDataStore(
        filename, fields=fields, fix_cf=fix_cf,
        tracerinfo_file=tracerinfo_file,
        diaginfo_file=diaginfo_file, endian=endian,
        default_dtype=default_dtype
    )

    ds = xr.Dataset.load_store(store)

    # ds = xr.decode_cf(ds)

    return ds


# def _get_bpch_contents(datablocks):
#     """ Generate sufficient information to create separate Datasets for each
#     field and timeslice in a given BPCH output dataset. """
#
#     for datablock in datablocks:
#
#         vname = datablock['category'] + "_" + datablock['name']
#
#         # Create a new variable if it's not already present
#         tracerinfo = datablock['tracerinfo']
#         attrs = dict(
#             long_name=tracerinfo['full_name'],
#             moulecular_weight=tracerinfo['molecular_weight'],
#             scale=tracerinfo['scale'],
#             units=tracerinfo['unit']
#         )
#
#         shape, origin, times = \
#             datablock['shape'], datablock['origin'], datablock['times']
#
#         if len(shape) == 2:
#             dims = ['time', 'lon', 'lat', ]
#         else:
#             dims = ['time', 'lon', 'lat', 'lev', ]
#
#         # Don't immediately load... let it be lazy!
#         # # TODO: using 'chunks' parameter, wrap this in a dask. delayed() call
#         # if vname in self._variables:
#         #     v = self._variables[vname]
#         #     data = np.concatenate(
#         #         [v.data, datablock['data'][np.newaxis, ...]], axis=0
#         #     )
#         # else:
#         # TODO: Need a better way to concatenate the variable data; doing it this way incurs a huge penalty because everything has to be read from the disk.
#         data = datablock['data'][None]
#
#         var = xr.Variable(dims, data, attrs)
#         # if vname in self._variables:
#         #     var = self._variables[vname].concat(var)
#
#         self._variables[vname] = var




class _BPCHDataStore(xr.backends.common.AbstractDataStore):
    """ Backend for representing bpch binary output. """

    def __init__(self, filename, fields=[], fix_cf=True,
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
            dummy_prefix_dims=1, concat_blocks=True,
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

            var = xr.Variable(dims, data, attrs)
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
            {'bounds': 'time_bnds'}
        )
        self._variables['time_bnds'] = xr.Variable(
            ['time', 'nv'], self._time_bnds, {}
        )

    def load_from_datablocks(self, datablocks, fields=[]):

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
        Compute the grid diemsnsions for a given datablock.

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
        return self._attributes

    def get_dimensions(self):
        return self._dimensions

    def close(self):
        pass
