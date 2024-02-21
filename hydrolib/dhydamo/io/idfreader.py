"""
Functions for reading and writing iMOD Data Files (IDFs) to ``xarray`` objects.

The primary functions to use are :func:`imod.idf.open` and
:func:`imod.idf.save`, though lower level functions are also available.

The function in this file are taken from the imod-python package version 0.10.1. 

The only functionality needed in dhydamo is the capability to read IDF files.
"""

import collections
import functools
import glob
import itertools
import pathlib
import re
import struct
import warnings
import cftime
import dask.array
import numpy as np
import dateutil.parser
import xarray as xr
import collections
import functools
import glob
import itertools
import pathlib

f_open = open

def _all_equal(seq, elem):
    """Raise an error if not all elements of a list are equal"""
    if not seq.count(seq[0]) == len(seq):
        raise ValueError(f"All {elem} must be the same, found: {set(seq)}")

def _check_cellsizes(cellsizes):
    """
    Checks if cellsizes match, raises ValueError otherwise

    Parameters
    ----------
    cellsizes : list of tuples
        tuples may contain:
        * two floats, dx and dy, for equidistant files
        * two ndarrays, dx and dy, for nonequidistant files

    Returns
    -------
    None
    """
    msg = "Cellsizes of IDFs do not match"
    if len(cellsizes) == 1:
        return None
    try:
        if not (cellsizes.count(cellsizes[0]) == len(cellsizes)):
            raise ValueError(msg)
    except ValueError:  # contains ndarrays
        try:
            # all ndarrays
            dx0, dy0 = cellsizes[0]
            for dx, dy in cellsizes[1:]:
                if np.allclose(dx0, dx) and np.allclose(dy0, dy):
                    pass
                else:
                    raise ValueError(msg)
        except ValueError:
            # some ndarrays, some floats
            # create floats for comparison with allclose
            try:
                dx = cellsizes[0][0][0]
                dy = cellsizes[0][1][0]
            except TypeError:
                dx = cellsizes[0][0]
                dy = cellsizes[0][1]
            # comparison
            for cellsize in cellsizes:
                # Unfortunately this allocates by broadcasting dx and dy
                if not np.allclose(cellsize[0], dx):
                    raise ValueError(msg)
                if not np.allclose(cellsize[1], dy):
                    raise ValueError(msg)

def _to_nan(a, nodata):
    """Change all nodata values in the array to NaN"""
    # it needs to be NaN for xarray to deal with it properly
    # no need to store the nodata value if it is always NaN
    if np.isnan(nodata):
        return a
    else:
        isnodata = np.isclose(a, nodata)
        a[isnodata] = np.nan
        return a

def _array_z_coord(coords, tops, bots, unique_indices):
    top = np.array(tops)[unique_indices]
    bot = np.array(bots)[unique_indices]
    dz = top - bot
    z = top - 0.5 * dz
    if top[0] > top[1]:  # decreasing coordinate
        dz *= -1.0
    if np.allclose(dz, dz[0]):
        coords["dz"] = dz[0]
    else:
        coords["dz"] = ("layer", dz)
    coords["z"] = ("layer", z)
    return coords


def _scalar_z_coord(coords, tops, bots):
    # They must be all the same to be used, as they cannot be assigned
    # to layer.
    top = np.unique(tops)
    bot = np.unique(bots)
    if top.size == bot.size == 1:
        dz = top - bot
        z = top - 0.5 * dz
        coords["dz"] = float(dz)  # cast from array
        coords["z"] = float(z)
    return coords


def _initialize_groupby(ndims):
    """
    This function generates a data structure consisting of defaultdicts, to use
    for grouping arrays by dimension. The number of dimensions may vary, so the
    degree of nesting might vary as well.

    For a single dimension such as layer, it'll look like:
    d = {1: da1, 2: da2, etc.}

    For two dimensions, layer and time:
    d = {"2001-01-01": {1: da1, 2: da3}, "2001-01-02": {1: da3, 2: da4}, etc.}

    And so on for more dimensions.

    Defaultdicts are very well suited to this application. The
    itertools.groupby object does not provide any benefits in this case, it
    simply provides a generator; its entries have to come presorted. It also
    does not provide tools for these kind of variably nested groupby's.

    Pandas.groupby does provide this functionality. However, pandas dataframes
    do not accept any field value, whereas these dictionaries do. Might be
    worthwhile to look into, if performance is an issue.

    Parameters
    ----------
    ndims : int
        Number of dimensions

    Returns
    -------
        groupby : Defaultdicts, ndims - 1 times nested
    """
    # In explicit form, say we have ndims=5
    # Then, writing it out, we get:
    # a = partial(defaultdict, {})
    # b = partial(defaultdict, a)
    # c = partial(defaultdict, b)
    # d = defaultdict(c)
    # This can obviously be done iteratively.
    if ndims == 1:
        return {}
    elif ndims == 2:
        return collections.defaultdict(dict)
    else:
        d = functools.partial(collections.defaultdict, dict)
        for _ in range(ndims - 2):
            d = functools.partial(collections.defaultdict, d)
        return collections.defaultdict(d)


def _set_nested(d, keys, value):
    """
    Set in the deepest dict of a set of nested dictionaries, as created by the
    _initialize_groupby function above.

    Mutates d.

    Parameters
    ----------
    d : (Nested dict of) dict
    keys : list of keys
        Each key is a level of nesting
    value : dask array, typically

    Returns
    -------
    None
    """
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        _set_nested(d[keys[0]], keys[1:], value)


def _sorteddict(d):
    """
    Sorts a variably nested dict (of dicts) by keys.

    Each dictionary will be sorted by its keys.

    Parameters
    ----------
    d : (Nested dict of) dict

    Returns
    -------
    sorted_lists : list (of lists)
        Values sorted by keys, matches the nesting of d.
    """
    firstkey = next(iter(d.keys()))
    if not isinstance(d[firstkey], dict):  # Base case
        return [v for (_, v) in sorted(d.items(), key=lambda t: t[0])]
    else:  # Recursive case
        return [_sorteddict(v) for (_, v) in sorted(d.items(), key=lambda t: t[0])]


def _ndconcat(arrays, ndim):
    """
    Parameters
    ----------
    arrays : list of lists, n levels deep.
        E.g.  [[da1, da2], [da3, da4]] for n = 2.
        (compare with docstring for _initialize_groupby)
    ndim : int
        number of dimensions over which to concatenate.

    Returns
    -------
    concatenated : xr.DataArray
        Input concatenated over n dimensions.
    """
    if ndim == 1:  # base case
        return dask.array.stack(arrays, axis=0)
    else:  # recursive case
        ndim = ndim - 1
        out = [_ndconcat(arrays_in, ndim) for arrays_in in arrays]
        return dask.array.stack(out, axis=0)



def _xycoords(bounds, cellsizes):
    """Based on bounds and cellsizes, construct coords with spatial information"""
    # unpack tuples
    xmin, xmax, ymin, ymax = bounds
    dx, dy = cellsizes
    coords = collections.OrderedDict()
    # from cell size to x and y coordinates
    if isinstance(dx, (int, float)):  # equidistant
        coords["x"] = np.arange(xmin + dx / 2.0, xmax, dx)
        coords["y"] = np.arange(ymax + dy / 2.0, ymin, dy)
        coords["dx"] = float(dx)
        coords["dy"] = float(dy)
    else:  # nonequidistant
        # even though IDF may store them as float32, we always convert them to float64
        dx = dx.astype(np.float64)
        dy = dy.astype(np.float64)
        coords["x"] = xmin + np.cumsum(dx) - 0.5 * dx
        coords["y"] = ymax + np.cumsum(dy) - 0.5 * dy
        if np.allclose(dx, dx[0]) and np.allclose(dy, dy[0]):
            coords["dx"] = float(dx[0])
            coords["dy"] = float(dy[0])
        else:
            coords["dx"] = ("x", dx)
            coords["dy"] = ("y", dy)
    return coords

def _convert_datetimes(times, use_cftime):
    """
    Return times as np.datetime64[ns] or cftime.DatetimeProlepticGregorian
    depending on whether the dates fall within the inclusive bounds of
    np.datetime64[ns]: [1678-01-01 AD, 2261-12-31 AD].

    Alternatively, always returns as cftime.DatetimeProlepticGregorian if
    ``use_cf_time`` is True.
    """
    if all(time == "steady-state" for time in times):
        return times, False

    out_of_bounds = False
    if use_cftime:
        converted = [
            cftime.DatetimeProlepticGregorian(*time.timetuple()[:6]) for time in times
        ]
    else:
        for time in times:
            year = time.year
            if year < 1678 or year > 2261:
                out_of_bounds = True
                break

        if out_of_bounds:
            use_cftime = True
            msg = "Dates are outside of np.datetime64[ns] timespan. Converting to cftime.DatetimeProlepticGregorian."
            warnings.warn(msg)
            converted = [
                cftime.DatetimeProlepticGregorian(*time.timetuple()[:6])
                for time in times
            ]
        else:
            converted = [np.datetime64(time, "ns") for time in times]

    return converted, use_cftime


def _dims_coordinates(header_coords, bounds, cellsizes, tops, bots, use_cftime):
    # create coordinates
    coords = _xycoords(bounds[0], cellsizes[0])
    dims = ["y", "x"]
    # Time and layer have to be special cased.
    # Time due to the multitude of datetimes possible
    # Layer because layer is required to properly assign top and bot data.
    haslayer = False
    hastime = False
    otherdims = []
    for dim in list(header_coords.keys()):
        if dim == "layer":
            haslayer = True
            coords["layer"], unique_indices = np.unique(
                header_coords["layer"], return_index=True
            )
        elif dim == "time":
            hastime = True
            times, use_cftime = _convert_datetimes(
                header_coords["time"], use_cftime
            )
            if use_cftime:
                coords["time"] = xr.CFTimeIndex(np.unique(times))
            else:
                coords["time"] = np.unique(times)
        else:
            otherdims.append(dim)
            coords[dim] = np.unique(header_coords[dim])

    # Ensure right dimension
    if haslayer:
        dims.insert(0, "layer")
    if hastime:
        dims.insert(0, "time")
    for dim in otherdims:
        dims.insert(0, dim)

    # Deal with voxel idf top and bottom data
    all_have_z = all(map(lambda v: v is not None, itertools.chain(tops, bots)))
    if all_have_z:
        if haslayer and coords["layer"].size > 1:
            coords = _array_z_coord(coords, tops, bots, unique_indices)
        else:
            coords = _scalar_z_coord(coords, tops, bots)

    return dims, coords


def _dask(path, attrs=None, pattern=None, _read=None, header=None):
    """
    Read a single IDF file to a dask.array

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be read
    attrs : dict, optional
        A dict as returned by imod.idf.header, this function is called if not supplied.
        Used to minimize unneeded filesystem calls.
    pattern : str, regex pattern, optional
        If the filenames do match default naming conventions of
        {name}_{time}_l{layer}, a custom pattern can be defined here either
        as a string, or as a compiled regular expression pattern. Please refer
        to the examples in ``imod.idf.open``.

    Returns
    -------
    dask.array
        A float32 dask.array with shape (nrow, ncol) of the values
        in the IDF file. On opening all nodata values are changed
        to NaN in the dask.array.
    dict
        A dict with all metadata.
    """

    path = pathlib.Path(path)

    if attrs is None:
        attrs = header(path, pattern)
    # If we don't unpack, it seems we run into trouble with the dask array later
    # on, probably because attrs isn't immutable. This works fine instead.
    headersize = attrs.pop("headersize")
    nrow = attrs["nrow"]
    ncol = attrs["ncol"]
    dtype = attrs["dtype"]
    # In case of floating point data, nodata is always represented by nan.
    if "float" in dtype:
        nodata = attrs.pop("nodata")
    else:
        nodata = attrs["nodata"]

    # Dask delayed caches the input arguments. If the working directory changes
    # before .compute(), the file cannot be found if the path is relative.
    abspath = path.resolve()
    # dask.delayed requires currying
    a = dask.delayed(_read)(abspath, headersize, nrow, ncol, nodata, dtype)
    x = dask.array.from_delayed(a, shape=(nrow, ncol), dtype=dtype)
    return x, attrs


def _load(paths, use_cftime, pattern, _read, header):
    """Combine a list of paths to IDFs to a single xarray.DataArray"""
    # this function also works for single IDFs

    headers = [header(p, pattern) for p in paths]
    names = [h["name"] for h in headers]
    _all_equal(names, "names")

    # Extract data from headers
    bounds = []
    cellsizes = []
    tops = []
    bots = []
    header_coords = collections.defaultdict(list)
    for h in headers:
        bounds.append((h["xmin"], h["xmax"], h["ymin"], h["ymax"]))
        cellsizes.append((h["dx"], h["dy"]))
        tops.append(h.get("top", None))
        bots.append(h.get("bot", None))
        for key in h["dims"]:
            header_coords[key].append(h[key])
    # Do a basic check whether IDFs align in x and y
    _all_equal(bounds, "bounding boxes")
    _check_cellsizes(cellsizes)
    # Generate coordinates
    dims, coords = _dims_coordinates(
        header_coords, bounds, cellsizes, tops, bots, use_cftime
    )
    # This part have makes use of recursion to deal with an arbitrary number
    # of dimensions. It may therefore be a little hard to read.
    groupbydims = dims[:-2]  # skip y and x
    ndim = len(groupbydims)
    groupby = _initialize_groupby(ndim)
    if ndim == 0:  # Single idf
        dask_array, _ = _dask(paths[0], headers[0], _read=_read)
    else:
        for path, attrs in zip(paths, headers):
            da, _ = _dask(path, attrs=attrs, _read=_read)
            groupbykeys = [attrs[k] for k in groupbydims]
            _set_nested(groupby, groupbykeys, da)
        dask_arrays = _sorteddict(groupby)
        dask_array = _ndconcat(dask_arrays, ndim)

    out = xr.DataArray(dask_array, coords, dims, name=names[0])

    first_attrs = headers[0]

    if "crs" in first_attrs:
        out.attrs["crs"] = first_attrs["crs"]
    if "nodata" in first_attrs:
        out.attrs["nodata"] = first_attrs["nodata"]

    return out

def _groupdict(stem, pattern):
    if pattern is not None:
        if isinstance(pattern, pattern):
            d = pattern.match(stem).groupdict()
        else:
            pattern = pattern.lower()
            # Get the variables between curly braces
            in_curly = re.compile(r"{(.*?)}").findall(pattern)
            regex_parts = {key: f"(?P<{key}>[\\w.-]+)" for key in in_curly}
            # Format the regex string, by filling in the variables
            simple_regex = pattern.format(**regex_parts)
            re_pattern = re.compile(simple_regex)
            # Use it to get the required variables
            d = re_pattern.match(stem).groupdict()
    else:  # Default to "iMOD conventions": {name}_c{species}_{time}_l{layer}
        has_layer = bool(re.search(r"_l\d+$", stem))
        has_species = bool(
            re.search(r"conc_c\d{1,3}_\d{8,14}", stem)
        )  # We are strict in recognizing species
        try:  # try for time
            base_pattern = r"(?P<name>[\w-]+)"
            if has_species:
                base_pattern += r"_c(?P<species>[0-9]+)"
            base_pattern += r"_(?P<time>[0-9-]{6,})"
            if has_layer:
                base_pattern += r"_l(?P<layer>[0-9]+)"
            re_pattern = re.compile(base_pattern)
            d = re_pattern.match(stem).groupdict()
        except AttributeError:  # probably no time
            base_pattern = r"(?P<name>[\w-]+)"
            if has_species:
                base_pattern += r"_c(?P<species>[0-9]+)"
            if has_layer:
                base_pattern += r"_l(?P<layer>[0-9]+)"
            re_pattern = re.compile(base_pattern)
            d = re_pattern.match(stem).groupdict()
    return d

def to_datetime(s):
    # try:
    #     time = datetime.datetime.strptime(s, DATETIME_FORMATS[len(s)])
    # except (ValueError, KeyError):  # Try fullblown dateutil date parser
    time = dateutil.parser.parse(s)
    return time


def decompose(path, pattern=None):
    r"""
    Parse a path, returning a dict of the parts, following the iMOD conventions.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file. Upper case is ignored.
    pattern : str, regex pattern, optional
        If the path is not made up of standard paths, and the default decompose
        does not produce the right result, specify the used pattern here. See
        the examples below.

    Returns
    -------
    d : dict
        Dictionary with name of variable and dimensions

    Examples
    --------
    Decompose a path, relying on default conventions:

    >>> decompose("head_20010101_l1.idf")

    Do the same, by specifying a format string pattern, excluding extension:

    >>> decompose("head_20010101_l1.idf", pattern="{name}_{time}_l{layer}")

    This supports an arbitrary number of variables:

    >>> decompose("head_slr_20010101_l1.idf", pattern="{name}_{scenario}_{time}_l{layer}")

    The format string pattern will only work on tidy paths, where variables are
    separated by underscores. You can also pass a compiled regex pattern.
    Make sure to include the ``re.IGNORECASE`` flag since all paths are lowered.

    >>> import re
    >>> pattern = re.compile(r"(?P<name>[\w]+)L(?P<layer>[\d+]*)")
    >>> decompose("headL11", pattern=pattern)

    However, this requires constructing regular expressions, which is generally
    a fiddly process. The website https://regex101.com is a nice help.
    Alternatively, the most pragmatic solution may be to just rename your files.
    """
    path = pathlib.Path(path)
    # We'll ignore upper case
    stem = path.stem.lower()

    d = _groupdict(stem, pattern)
    dims = list(d.keys())
    # If name is not provided, generate one from other fields
    if "name" not in d.keys():
        d["name"] = "_".join(d.values())
    else:
        dims.remove("name")

    # TODO: figure out what to with user specified variables
    # basically type inferencing via regex?
    # if purely numerical \d* -> int or float
    #    if \d*\.\d* -> float
    # else: keep as string

    # String -> type conversion
    if "layer" in d.keys():
        d["layer"] = int(d["layer"])
    if "species" in d.keys():
        d["species"] = int(d["species"])
    if "time" in d.keys():
        d["time"] = to_datetime(d["time"])
    if "steady-state" in d["name"]:
        # steady-state as time identifier isn't picked up by <time>[0-9] regex
        d["name"] = d["name"].replace("_steady-state", "")
        d["time"] = "steady-state"
        dims.append("time")

    d["extension"] = path.suffix
    d["directory"] = path.parent
    d["dims"] = dims
    return d



def header(path, pattern):
    """Read the IDF header information into a dictionary"""
    attrs = decompose(path, pattern)
    with f_open(path, "rb") as f:
        reclen_id = struct.unpack("i", f.read(4))[0]  # Lahey RecordLength Ident.
        if reclen_id == 1271:
            floatsize = intsize = 4
            floatformat = "f"
            intformat = "i"
            dtype = "float32"
            doubleprecision = False
        # 2296 was a typo in the iMOD manual. Keep 2296 around in case some IDFs
        # were written with this identifier to avoid possible incompatibility
        # issues.
        elif reclen_id == 2295 or reclen_id == 2296:
            floatsize = intsize = 8
            floatformat = "d"
            intformat = "q"
            dtype = "float64"
            doubleprecision = True
        else:
            raise ValueError(
                f"Not a supported IDF file: {path}\n"
                "Record length identifier should be 1271 or 2295, "
                f"received {reclen_id} instead."
            )

        # Header is fully doubled in size in case of double precision ...
        # This means integers are also turned into 8 bytes
        # and requires padding with some additional bytes
        if doubleprecision:
            f.read(4)  # not used

        ncol = struct.unpack(intformat, f.read(intsize))[0]
        nrow = struct.unpack(intformat, f.read(intsize))[0]
        attrs["xmin"] = struct.unpack(floatformat, f.read(floatsize))[0]
        attrs["xmax"] = struct.unpack(floatformat, f.read(floatsize))[0]
        attrs["ymin"] = struct.unpack(floatformat, f.read(floatsize))[0]
        attrs["ymax"] = struct.unpack(floatformat, f.read(floatsize))[0]
        # dmin and dmax are recomputed during writing
        f.read(floatsize)  # dmin, minimum data value present
        f.read(floatsize)  # dmax, maximum data value present
        nodata = struct.unpack(floatformat, f.read(floatsize))[0]
        attrs["nodata"] = nodata
        # flip definition here such that True means equidistant
        # equidistant IDFs
        ieq = not struct.unpack("?", f.read(1))[0]
        itb = struct.unpack("?", f.read(1))[0]

        f.read(2)  # not used
        if doubleprecision:
            f.read(4)  # not used

        if ieq:
            # dx and dy are stored positively in the IDF
            # dy is made negative here to be consistent with the nonequidistant case
            attrs["dx"] = struct.unpack(floatformat, f.read(floatsize))[0]
            attrs["dy"] = -struct.unpack(floatformat, f.read(floatsize))[0]

        if itb:
            attrs["top"] = struct.unpack(floatformat, f.read(floatsize))[0]
            attrs["bot"] = struct.unpack(floatformat, f.read(floatsize))[0]

        if not ieq:
            # dx and dy are stored positive in the IDF, but since the difference between
            # successive y coordinates is negative, it is made negative here
            attrs["dx"] = np.fromfile(f, dtype, ncol)
            attrs["dy"] = -np.fromfile(f, dtype, nrow)

        # These are derived, remove after using them downstream
        attrs["headersize"] = f.tell()
        attrs["ncol"] = ncol
        attrs["nrow"] = nrow
        attrs["dtype"] = dtype

    return attrs


def _read(path, headersize, nrow, ncol, nodata, dtype):
    """
    Read a single IDF file to a numpy.ndarray

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be read
    headersize : int
        byte size of header
    nrow : int
    ncol : int
    nodata : np.float

    Returns
    -------
    numpy.ndarray
        A float numpy.ndarray with shape (nrow, ncol) of the values
        in the IDF file. On opening all nodata values are changed
        to NaN in the numpy.ndarray.
    """
    with f_open(path, "rb") as f:
        f.seek(headersize)
        a = np.reshape(np.fromfile(f, dtype, nrow * ncol), (nrow, ncol))
    return _to_nan(a, nodata)


def read(path, pattern=None):
    """
    Read a single IDF file to a numpy.ndarray

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be read
    pattern : str, regex pattern, optional
        If the filenames do match default naming conventions of
        {name}_{time}_l{layer}, a custom pattern can be defined here either
        as a string, or as a compiled regular expression pattern. Please refer
        to the examples for ``imod.idf.open``.

    Returns
    -------
    numpy.ndarray
        A float numpy.ndarray with shape (nrow, ncol) of the values
        in the IDF file. On opening all nodata values are changed
        to NaN in the numpy.ndarray.
    dict
        A dict with all metadata.
    """
    warnings.warn(
        "The idf.read() function is deprecated. To get a numpy array of an IDF, "
        "use instead: imod.idf.open(path).values",
        FutureWarning,
    )
    attrs = header(path, pattern)
    headersize = attrs.pop("headersize")
    nrow = attrs.pop("nrow")
    ncol = attrs.pop("ncol")
    nodata = attrs.pop("nodata")
    dtype = attrs.pop("dtype")
    return _read(path, headersize, nrow, ncol, nodata, dtype), attrs


# Open IDFs for multiple times and/or layers into one DataArray
def open(path, use_cftime=False, pattern=None):
    r"""
    Open one or more IDF files as an xarray.DataArray.

    In accordance with xarray's design, ``open`` loads the data of IDF files
    lazily. This means the data of the IDFs are not loaded into memory until the
    data is needed. This allows for easier handling of large datasets, and
    more efficient computations.

    Parameters
    ----------
    path : str, Path or list
        This can be a single file, 'head_l1.idf', a glob pattern expansion,
        'head_l*.idf', or a list of files, ['head_l1.idf', 'head_l2.idf'].
        Note that each file needs to be of the same name (part before the
        first underscore) but have a different layer and/or timestamp,
        such that they can be combined in a single xarray.DataArray.
    use_cftime : bool, optional
        Use ``cftime.DatetimeProlepticGregorian`` instead of `np.datetime64[ns]`
        for the time axis.

        Dates are normally encoded as ``np.datetime64[ns]``; however, if dates
        fall before 1678 or after 2261, they are automatically encoded as
        ``cftime.DatetimeProlepticGregorian`` objects rather than
        ``np.datetime64[ns]``.
    pattern : str, regex pattern, optional
        If the filenames do match default naming conventions of
        {name}_{time}_l{layer}, a custom pattern can be defined here either
        as a string, or as a compiled regular expression pattern. See the
        examples below.

    Returns
    -------
    xarray.DataArray
        A float xarray.DataArray of the values in the IDF file(s).
        All metadata needed for writing the file to IDF or other formats
        using imod.rasterio are included in the xarray.DataArray.attrs.

    Examples
    --------
    Open an IDF file:

    >>> da = imod.idf.open("example.idf")

    Open an IDF file, relying on default naming conventions to identify
    layer:

    >>> da = imod.idf.open("example_l1.idf")

    Open an IDF file, relying on default naming conventions to identify layer
    and time:

    >>> head = imod.idf.open("head_20010101_l1.idf")

    Open multiple IDF files, in this case files for the year 2001 for all
    layers, again relying on default conventions for naming:

    >>> head = imod.idf.open("head_2001*_l*.idf")

    The same, this time explicitly specifying ``name``, ``time``, and ``layer``:

    >>> head = imod.idf.open("head_2001*_l*.idf", pattern="{name}_{time}_l{layer}")

    The format string pattern will only work on tidy paths, where variables are
    separated by underscores. You can also pass a compiled regex pattern.
    Make sure to include the ``re.IGNORECASE`` flag since all paths are lowered.

    >>> import re
    >>> pattern = re.compile(r"(?P<name>[\w]+)L(?P<layer>[\d+]*)", re.IGNORECASE)
    >>> head = imod.idf.open("headL11", pattern=pattern)

    However, this requires constructing regular expressions, which is
    generally a fiddly process. Regex notation is also impossible to
    remember. The website https://regex101.com is a nice help. Alternatively,
    the most pragmatic solution may be to just rename your files.
    """

    if isinstance(path, list):
        return _load(path, use_cftime, pattern, _read, header)
    elif isinstance(path, pathlib.Path):
        path = str(path)

    paths = [pathlib.Path(p) for p in glob.glob(path)]
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"Could not find any files matching {path}")
    return _load(paths, use_cftime, pattern, _read, header)

