"""General helper and utility functions."""
from datetime import datetime, timezone
from typing import Union

import numpy as np
import pandas as pd
from bidict import bidict


def get_tc3_dtypes():
    """List of conversion from twincat to (numpy, pandas) dtypes."""
    import numpy as np

    tc3_dtypes = {
        "BIT": (np.bool_, "boolean"),
        # "BOOL": (np.bool_, "boolean"),
        # "BIT8": (np.bool_, "UInt8"),
        "INT8": (np.int8, "Int8"),
        "INT16": (np.int16, "Int16"),
        "INT32": (np.int32, "Int32"),
        "INT64": (np.int64, "Int64"),
        "UINT8": (np.uint8, "UInt8"),
        "UINT16": (np.uint16, "UInt16"),
        "UINT32": (np.uint32, "UInt32"),
        "UINT64": (np.uint64, "UInt64"),
        "REAL32": (np.float32, np.float32),
        "REAL64": (np.float64, np.float64),
        # "BIT_ARRAY_8": (, ),
        # "STRING_255": (, ),
        # "IMAGE": (, ),
    }
    return tc3_dtypes


# list of channel metadata keys
_channel_meta_keys = {
    "SymbolComment": "symbol_comment",
    "DataType": "data_type",
    "SampleTime": "sample_time",
    "VariableSize": "variable_size",
    "SymbolBased": "symbol_based",
    "IndexGroup": "index_group",
    "IndexOffset": "index_offset",
    "SymbolName": "symbol_name",
    "NetID": "net_id",
    "Port": "port",
    "Offset": "offset",
    "ScaleFactor": "scale_factor",
    "BitMask": "bit_mask",
    "Unit": "unit",
    "UnitScaleFactor": "unit_scale_factor",
    "UnitOffset": "unit_offset",
}
_channel_meta_keys = bidict(_channel_meta_keys)

# Offset from filetime-Origin to 1970-1-1 00.00:00 in MS filetime units
MS_FILETIME_OFFSET = int(116444736000000000)


def filetime_to_dt(ft: int) -> datetime:
    """
    Convert a Microsoft filetime number to a Python datetime in UTC time.

    >>> filetime_to_dt(116444736000000000)
    datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)

    >>> filetime_to_dt(128930364000000000)
    datetime.datetime(2009, 7, 25, 23, 0, tzinfo=datetime.timezone.utc)

    >>> filetime_to_dt(128930364000001000)
    datetime.datetime(2009, 7, 25, 23, 0, 0, 100, tzinfo=datetime.timezone.utc)
    """
    return datetime.fromtimestamp((ft - MS_FILETIME_OFFSET) / 1e7, tz=timezone.utc)


def parse_unit_string(tc3_unit_string: str) -> Union[str, None]:
    """
    Extract unit symbol from unit string imported from TwinCAT3 scope file.

    Parameters
    ----------
    tc3_unit_string: str
        The unit string in TwinCAT format.

    Returns
    -------
    unit : str
         string containing only the unit
    """
    if not tc3_unit_string:
        return None

    if tc3_unit_string.split(" ")[0] == "(None)":
        return None
    else:
        return tc3_unit_string.split(" ")[0]


def to_datetime_from_ms(t, origin) -> pd.DatetimeIndex:
    """
    Fast conversion from time-array in ms (like TC3 data) to UTC pandas.DatetimeIndex .

    Parameters
    ----------
    t : array-like
        timestamps from origin in ms
    origin : timestamp-like
        if True, convert DataFrame index to absolute UTC timestamps

    Returns
    -------
    dt : pandas.DatetimeIndex
         time naive DatetimeIndex (in UTC reference)
    """
    t_int = (t * 1e6).astype(np.int64)
    origin_ts = pd.Timestamp(origin).replace(tzinfo=timezone.utc).tz_convert(None)
    origin_int = origin_ts.to_datetime64().astype(np.int64)
    return pd.to_datetime(t_int + origin_int)


def make_time_index(
    values: np.ndarray,
    timestamps: bool,
    start_time: datetime,
    tz: Union[str, None] = None,
) -> Union[pd.TimedeltaIndex, pd.DatetimeIndex]:
    """Convert time values to pandas time index."""
    if timestamps:
        return to_datetime_from_ms(values, start_time).tz_convert(tz)
    return pd.TimedeltaIndex((values * 1e6).astype(np.int64))
