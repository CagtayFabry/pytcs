"""pytcs - Python API for reading TwinCAT Scope Files."""
from __future__ import annotations

import gzip
import importlib
from collections.abc import ItemsView, Iterator, ValuesView
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO, IOBase, StringIO
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, KeysView, Union
from warnings import warn

import numpy as np
import pandas as pd

from pytcs.helpers import (
    filetime_to_dt,
    get_tc3_dtypes,
    parse_unit_string,
    to_datetime_from_ms,
)

if TYPE_CHECKING:
    from datetime import timedelta

    import xarray as xr


@dataclass
class ScopeFileInfo:
    """General ScopeFile metadata."""

    scope_name: str = None
    start_time: datetime = None
    end_time: datetime = None
    start_time_tc3: int = None
    end_time_tc3: int = None


@dataclass
class ScopeChannel:
    """Class representing a single TwinCAT Measurement channel in a file."""

    name: str
    time_col: int = field(repr=False)
    value_col: int = field(repr=False)
    time: np.ndarray = None
    values: np.ndarray = None
    info: dict = field(default_factory=dict, repr=False)
    sample_time: float = field(default=None, init=False)
    time_offset: float = field(default=None, init=False)
    units: str = field(default=None, init=False)

    def update_attrs(self, decimal):
        """Update the class infos from the info dictionary."""
        self.sample_time: str = self.info.get("SampleTime")  # legacy
        self.sample_time: str = self.info.get("SampleTime[ms]")
        if self.sample_time:
            self.sample_time = float(self.sample_time.replace(decimal, "."))

        self.units = parse_unit_string(self.info.get("Unit"))

    def as_dict(self) -> dict:
        """Return a dict representation of the ScopeChannel."""
        channel = self.__dict__.copy()
        channel["values"] = channel.pop("_values")
        return channel

    def as_pandas(self) -> pd.Series:
        """Channel as pandas series."""
        s = pd.Series(data=self.values, index=self.time, name=self.name)
        return s.rename_axis("time")

    def as_dataarray(self) -> xr.DataArray:
        """Channel as xarray dataarray."""
        import xarray as xr

        if not (lt := len(self.time)) == (lv := len(self.values)):
            warn(
                f"Mismatch in time and value length for channel {self.name}",
                UserWarning,
                stacklevel=2,
            )
            return xr.DataArray(
                data=self.values[: min(lt, lv)],
                dims=["time"],
                coords={"time": self.time[: min(lt, lv)]},
                name=self.name,
            )

        return xr.DataArray(data=self.values, dims=["time"], name=self.name)

    # custom setter from https://stackoverflow.com/a/61480946/11242411
    def get_values(self) -> np.ndarray:
        """Reimplement default value getter."""
        return self._values

    def set_values(self, values: np.ndarray):
        """Set custom value with simple length check.

        When assigning values to a scope channel, the length of the values are checked
        against the length of the time attribute.

        Parameters
        ----------
        values

        """
        if values is None:
            self._values = values
            return

        if self.time is None:
            raise ValueError("Cannot set values on 'ScopeChannel' without time.")

        if (values is not None) and (len(values) != len(self.time)):
            raise RuntimeError(
                f"length difference for channel '{self.name}': "
                f"time={len(self.time)}, values={len(values)}"
            )
        self._values = values

    def __str__(self) -> str:
        """Show simple text output."""
        s = f"<TwinCAT Scope Channel at {hex(id(self))}> "
        s += f"\nname:          {self.name}"
        s += "\nlength:        "
        if self.values is None:
            s += "(unloaded)"
        else:
            s += f"{len(self.values)}"
        s += f"\nsample time:   {self.sample_time} ms"
        s += f"\nunits:         {self.units}"
        return s


# https://github.com/florimondmanca/www/issues/102#issuecomment-733947821
ScopeChannel.values = property(ScopeChannel.get_values, ScopeChannel.set_values)


class ScopeFile:
    """Class for reading TwinCAT Scope files."""

    def __init__(
        self,
        filepath_or_buffer: Union[Path, str, StringIO, BytesIO],
        delimiter: str = None,
        decimal: str = None,
        encoding: str = "utf-8",
        compression: str = "infer",
    ):
        """Open a TwinCAT Scope file.

        Parameters
        ----------
        filepath_or_buffer
            The file or buffer to open.
            TwinCAT Scope CSV style files are supported.
            Files can be gzipped (e.g. .csv.gz).
        delimiter
            The column delimiter used in the file.
            If no explicit value is given this is inferred form the file header.
        decimal
            The decimal delimiter used in the file.
            If no explicit value is given this is inferred form the file header.
        encoding
            File encoding (default: 'utf-8')
        compression
            File or data compression (default: 'infer' / autodetect)

        """
        self._delimiter: str = delimiter
        self._decimal: str = decimal

        self._file = filepath_or_buffer  # file handle, buffer etc.
        self._encoding: str = encoding
        self._compression = compression
        self._meta: dict = {}
        self.start_time: datetime = None
        self.run_time: timedelta = None

        self._channels: dict[str, ScopeChannel] = {}

        self._data: dict[int, np.ndarray] = {}  # column data

        # determine correct way to open file
        if isinstance(filepath_or_buffer, StringIO):
            self._read_header(filepath_or_buffer)
            return

        if isinstance(filepath_or_buffer, BytesIO) or Path(
            filepath_or_buffer
        ).suffix in [".gz", ".gzip"]:
            fopen = gzip.open
        else:
            fopen = open

        with fopen(self._file, "rt", encoding=self._encoding) as f:
            self._read_header(f)

    def load(self, channels: list[str] = None, native_dtypes=False, backend="pandas"):
        """Load one or more channels into memory.

        Parameters
        ----------
        channels
            List of channels to load from the file.
            If None is passed (the default), all channels will be loaded.
            Channels that are already loaded will be ignored and not be reloaded !
        native_dtypes
            If ``True``, convert the array values to native their numpy dtypes.
            Default: ``False``
        backend
            The CSV backend to use.
            Available backends are ``pandas`` (the default) and ``datatable``.
            Reading with pandas uses the ``engine=c``.
            For larger files, the ``datatable`` backend can be faster.
            The ``datatable`` backend is considered experimental and has known bugs with
            ``datatable<1.1.0`` for some CSV formats.

        """
        usecols = self._get_usecols(channels)

        if not usecols:  # already loaded
            return

        if backend == "pandas":
            data_dict = self._read_pandas(usecols, native_dtypes)
        elif backend == "pyarrow":
            data_dict = self._read_pyarrow(usecols, native_dtypes)
        elif backend == "datatable":
            if native_dtypes:
                warn(
                    "Ignoring option 'native_dtypes' with datatable backend.",
                    UserWarning,
                    stacklevel=2,
                )
            data_dict = self._read_datatable(usecols)
        else:
            raise ValueError(f"Unknown CSV backend: '{backend}'.")

        self._data.update(data_dict)
        self._update_time_links()

        self._update_data_refs()

    def as_dict(self) -> dict:
        """Convert scope file into regular Python dict."""
        sf_dict = {
            "scope_name": self._meta["ScopeName"],
            "file": self._meta["File"],
            "start_time": self.start_time,
            "run_time": self.run_time,
            "channels": {k: v.as_dict() for k, v in self.items()},
        }
        return sf_dict

    def as_pandas(
        self,
        channels: list[str] = None,
        time_fmt: str = "timestamp",
    ) -> pd.DataFrame:
        """Convert scope file into `pandas.DataFrame`.

        Parameters
        ----------
        channels
            List of channels to include in the dataset.
            Passing `None` (the default) will load and include all channels.
        time_fmt
            The dtype of the time coordinate.
            Either 'timestamp' for ``datetime64[ns]`` (the default) or 'timedelta' for
            ``timedelta64[ns]`` referenced to 'start_time'.

        """
        if time_fmt not in ["timestamp", "timedelta"]:
            raise ValueError(
                f"Unsupported option time_fmt='{time_fmt}'."
                f"Use 'timestamp' or 'timedelta'."
            )

        self.load(channels)

        _l = {}
        for t_col in set(self._time_mapping.values()):
            _min = [
                len(c.values)
                for n, c in self.items()
                if (self._time_mapping[c.time_col] == t_col) and (c.values is not None)
            ]
            _min.append(len(self._data[t_col]))
            _l[t_col] = min(_min)

        _dict = {
            t_col: pd.DataFrame(index=self._data[t_col][: _l[t_col]]).rename_axis(
                "time"
            )
            for t_col in set(self._time_mapping.values())
        }

        channels = self._get_channels(channels)
        for n in channels:
            c = self[n]
            t_col = self._time_mapping[c.time_col]

            _dict[t_col] = _dict[t_col].assign(**{c.name: c.values[: _l[t_col]]})

        _list = [v for v in _dict.values() if not v.empty]

        df = pd.DataFrame().join(_list, how="outer", sort=True)

        if time_fmt == "timedelta":
            df.index = pd.TimedeltaIndex((df.index * 1e6).astype(np.int64))
        else:
            df.index = to_datetime_from_ms(df.index, self.start_time)

        return df

    def as_xarray(
        self,
        channels: list[str] = None,
        time_fmt: str = "timestamp",
    ) -> xr.Dataset:
        """Convert scope file into `xarray.Dataset`.

        Important: Since xarray does not support timezone information in indexes, all
        times are given in UTC time when selecting ``time_fmt = 'timestamp'``

        Parameters
        ----------
        channels
            List of channels to include in the dataset.
            Passing `None` (the default) will load and include all channels.
        time_fmt
            The dtype of the time coordinate.
            Either 'timestamp' for ``datetime64[ns]`` (the default) or 'timedelta' for
            ``timedelta64[ns]`` referenced to 'start_time'.


        Returns
        -------
        xr.Dataset
            xarray Dataset with each channel as a variable.

        """
        import xarray as xr

        if time_fmt not in ["timestamp", "timedelta"]:
            raise ValueError(
                f"Unsupported option time_fmt='{time_fmt}'."
                f"Use 'timestamp' or 'timedelta'."
            )

        df = self.as_pandas(channels=channels, time_fmt=time_fmt)
        ds = xr.Dataset.from_dataframe(df)

        # assign metadata attributes
        ds.attrs = self._meta
        ds.attrs["start_time"] = self.start_time.isoformat()
        ds.attrs["run_time"] = str(self.run_time)
        for c in ds:
            ds[c].attrs = self[c].info
            ds[c].attrs["units"] = parse_unit_string(ds[c].attrs.get("Unit"))
        return ds

    def __repr__(self):
        """Show simple text output."""
        s = f"<TwinCAT Scope File at {hex(id(self))}> "
        s += f'\nname:    {self._meta["ScopeName"]}'
        s += f"\nstart:   {self.start_time.isoformat()}"
        s += f"\nruntime: {self.run_time}"

        s += "\n\nChannels:"
        for v in self._channels.values():
            unloaded = "*" if v.values is None else ""
            s += f"\n  {unloaded}{v.name}: {v.sample_time} ms [{v.units}]"
        return s

    def to_native_dtypes(self):
        """Convert values to their native numpy dtypes.

        The conversion is done inplace.
        """
        tc3_dtypes = get_tc3_dtypes()

        for c in self:
            self._data[self[c].value_col] = self._data[self[c].value_col].astype(
                tc3_dtypes[self[c].info.get("Data-Type", "REAL64")][0]
            )
            self[c].values = self._data[self[c].value_col]

    def __getitem__(self, item):
        """Get item from list of channels."""
        return self._channels[item]

    def __iter__(self) -> Iterator[str]:
        """Iterate over channels."""
        return iter(self._channels)

    def items(self) -> ItemsView:
        """Return channel dictionary ItemView."""
        return self._channels.items()

    def values(self) -> ValuesView[ScopeChannel]:
        """Return channel dictionary ValuesView."""
        return self._channels.values()

    def _read_header(self, f):
        """Read header from opened file.

        f should point to the beginning of the file.
        This will advance the file pointer to the first data line.

        - auto-detect file delimiter
        - populate file metadata
        - populate channel metadata
        - auto-detect decimal sign
        - build the time column mapping

        Parameters
        ----------
        f
            The opened stream like object.
            Must point to the beginning of the scope file.
        """
        # read and parse file info
        line = f.readline()
        if self._delimiter is None:
            self._delimiter = line[4]
        delimiter = self._delimiter

        if delimiter in [",", " "]:
            raise ValueError("Parsing of COMMA or SPACE delimited files not available.")

        self._meta["ScopeName"] = line.split(delimiter, maxsplit=1)[-1].rstrip()
        self._meta["File"] = f.readline().split(delimiter, maxsplit=1)[-1].rstrip()
        self._meta["StartTime"] = int(f.readline().split(delimiter)[1])
        self._meta["EndTime"] = int(f.readline().split(delimiter)[1])

        self.start_time = filetime_to_dt(self._meta["StartTime"])
        self.run_time = filetime_to_dt(self._meta["EndTime"]) - self.start_time

        line = f.readline()
        while line:
            if line.startswith("Name"):
                break
            line = f.readline()

        self._parse_metadata_block(f, line)

        # look for start of data block
        data_block_search_index = f.tell()
        line = f.readline()
        while line:
            if line[0].isnumeric():  # found first data line
                break
            data_block_search_index = f.tell()
            line = f.readline()

        # catch support for legacy files that end with delimiter
        self._line_end_delimiter = 1 if delimiter in line[-2:] else 0

        # try to auto detect decimal
        if not self._decimal:
            self._decimal = self._get_decimal_from_line(line, delimiter)
            while not self._decimal:
                self._decimal = self._get_decimal_from_line(f.readline(), delimiter)

        # determine the number of header rows
        f.seek(0)
        if isinstance(f, StringIO):  # we navigate characters
            _header = f.read(data_block_search_index)
            self._header_lines = _header.count("\n")
        else:
            _header = f.buffer.read(
                data_block_search_index
            )  # read to binary position from buffer
            self._header_lines = _header.count(b"\n")

        # build the time column mappings
        self._build_time_mapping(f.readline(), f.readline())

        # cleanup channels without name (legacy file exports)
        self._channels = {k: v for k, v in self._channels.items() if v.name}

        # update channel info
        for c in self.values():
            c.update_attrs(decimal=",")
            c.time_offset = self._time_offset[c.time_col]
            if not c.sample_time:
                c.sample_time = self._time_sample_time[c.time_col]

    def _parse_metadata_block(self, f, name_line):
        """Read the channel metadata block and add info to channels.

        Parameters
        ----------
        f
            The opened stream like object.
            Must point to the beginning of a line of channel metadata.
        name_line
            The line with channel names.

        """
        delimiter = self._delimiter

        channel_list = [
            sub[:-1].split(delimiter) for sub in name_line.split("Name" + delimiter)[1:]
        ]
        expected_fmt = [len(channels) for channels in channel_list]

        channels = []
        time_index = 0
        for sub_list in channel_list:
            for n, c in enumerate(sub_list):
                channels.append((time_index, time_index + n + 1, c))
            time_index += n + 2

            self._channels = {
                n: ScopeChannel(
                    name=n,
                    time_col=i1,
                    value_col=i2,
                )
                for i1, i2, n in channels
            }

        line = f.readline()
        while line:
            # TODO: fix possible linebreaks here

            if line == "\n":
                break

            self._read_meta_line(line, delimiter, expected_fmt)
            line = f.readline()

    def _build_time_mapping(self, line_0, line_1):
        """Build the mapping between channels and associated time columns.

        The '_time_mapping' is a shortend mapping assuming that channels with the same
        start time (offset) and sample time have he same resulting time columns (as they
        should).
        However, some TwinCAT bugs have been reported where time columns contain more
        values than expected. In this case the approach here can produce errors.

        This function also determines the start time and sample rates of all channels
        from the first two data rows.

        Parameters
        ----------
        line_0
        line_1

        """
        delimiter = self._delimiter
        decimal = self._decimal

        time_indx = list(dict.fromkeys(self._get_time_cols()))
        t_0 = [float(n) for n in line_0.replace(decimal, ".").rstrip().split(delimiter)]
        t_0 = [t_0[i] for i in time_indx]
        t_1 = [float(n) for n in line_1.replace(decimal, ".").rstrip().split(delimiter)]
        t_1 = [t_1[i] for i in time_indx]

        _time_meta = [(a, b) for a, b in zip(t_0, t_1)]
        self._time_mapping = {
            c: time_indx[_time_meta.index(_time_meta[i])]
            for i, c in enumerate(time_indx)
        }
        self._time_offset = {c: _time_meta[i][0] for i, c in enumerate(time_indx)}
        self._time_sample_time = {c: _time_meta[i][1] for i, c in enumerate(time_indx)}

    def _read_pandas(
        self, usecols: list[int], native_dtypes: bool
    ) -> dict[str, np.ndarray]:
        """Read data into dictionary using the pandas backend."""
        if isinstance(self._file, IOBase):  # read open streams from beginning
            self._file.seek(0)

        compression = self._compression
        if isinstance(self._file, BytesIO):  # assume gzip
            compression = "gzip"

        df = pd.read_csv(
            self._file,
            delimiter=self._delimiter,
            skiprows=self._header_lines,
            decimal=self._decimal,
            header=None,
            usecols=usecols,
            names=self._get_cols(),
            index_col=False,
            encoding=self._encoding,
            na_values=[" ", "EOF"],
            skip_blank_lines=True,
            engine="c",
            low_memory=False,
            compression=compression,
        )

        if native_dtypes:
            tc3_dtypes = get_tc3_dtypes()
            dtypes_np = {
                v.value_col: tc3_dtypes[v.info.get("Data-Type", "REAL64")][0]
                for v in self._channels.values()
            }
            dtypes_times = {k: np.float64 for k in self._get_time_cols()}
            dtypes_np.update(dtypes_times)
            data_dict = {k: df[k].dropna().to_numpy(dtypes_np[k]) for k in df}
        else:
            data_dict = {k: df[k].dropna().to_numpy(np.float64) for k in df}

        return data_dict

    def _read_pyarrow(
        self, usecols: list[int], native_dtypes: bool
    ) -> dict[str, np.ndarray]:
        """Read data into dictionary using the pyarrow backend of pandas."""
        from itertools import groupby

        if not importlib.util.find_spec("pyarrow"):
            warn(
                "'pyarrow' backend not found, using default pandas implementation.",
                UserWarning,
                stacklevel=2,
            )
            return self._read_pandas(usecols, native_dtypes)

        def all_equal(iterable):
            "Returns True if all the elements are equal to each other"
            g = groupby(iterable)
            return next(g, True) and not next(g, False)

        if usecols:
            warn(
                """Channel selection is not supported with 'pyarrow' backend, """
                """loading all channels.""",
                UserWarning,
                stacklevel=2,
            )
            usecols = None

        if not all_equal(self[c].sample_time for c in self):
            raise ValueError(
                "Unsupported file format for 'pyarrow' backend. (unequal sample times)"
            )

        if isinstance(self._file, IOBase):  # read open streams from beginning
            self._file.seek(0)

        compression = self._compression
        if isinstance(self._file, BytesIO):  # assume gzip
            compression = "gzip"

        df = pd.read_csv(
            self._file,
            delimiter=self._delimiter,
            skiprows=self._header_lines,
            # decimal=self._decimal, # unsupported with pyarrow
            header=None,
            usecols=usecols,
            names=self._get_cols(),
            # index_col=False,
            encoding=self._encoding,
            na_values=[" ", "EOF"],
            skip_blank_lines=True,
            engine="pyarrow",
            # low_memory=False, # unsupported with pyarrow
            compression=compression,
        )

        self._df = df

        if native_dtypes:
            tc3_dtypes = get_tc3_dtypes()
            dtypes_np = {
                v.value_col: tc3_dtypes[v.info.get("Data-Type", "REAL64")][0]
                for v in self._channels.values()
            }
            dtypes_times = {k: np.float64 for k in self._get_time_cols()}
            dtypes_np.update(dtypes_times)
            data_dict = {k: df[k].dropna().to_numpy(dtypes_np[k]) for k in df}
        else:
            data_dict = {k: df[k].dropna().to_numpy(np.float64) for k in df}

        return data_dict

    def _read_datatable(self, usecols: list[int]) -> dict[str, np.ndarray]:
        """Read data into dictionary using the datatable backend."""
        try:
            import datatable
        except ModuleNotFoundError:
            warn(
                "'datatable' backend not found, using pandas.",
                UserWarning,
                stacklevel=2,
            )
            return self._read_pandas(usecols)

        columns = [
            i in usecols
            for i in range(len(self._get_cols()) + self._line_end_delimiter)
        ]

        df = datatable.fread(
            self._file,
            header=False,
            columns=columns,
            skip_to_line=self._header_lines,
            fill=True,
            sep=self._delimiter,
            dec=self._decimal,
            skip_blank_lines=True,
            na_strings=["EOF"],
        )
        # self._df = df  # debug

        data_dict = {
            k: df[:, i].to_numpy(np.float64).squeeze() for i, k in enumerate(usecols)
        }
        data_dict = {k: v[~np.isnan(v)] for k, v in data_dict.items()}
        return data_dict

    def _update_time_links(self):
        """Update the time column data references."""
        for k, v in self._time_mapping.items():
            if (k not in self._data) & (v in self._data):
                self._data[k] = self._data[v]

    def _get_usecols(self, channels: list[str] = None) -> list[int]:
        """Get the column numbers to read from a list of column names.

        The column numbers include the corresponding time and data columns.
        Columns already loaded will be discarded.
        """
        if isinstance(channels, str):
            channels = [channels]
        if channels is None:
            channels = self._channels.keys()

        if diff := (set(channels) - self._channels.keys()):
            warn(
                f"Cannot find the following channels: {diff}", UserWarning, stacklevel=2
            )
            channels = self._channels.keys() - channels
        channels = list(channels)

        s = self._get_cols(channels) - self._data.keys()
        return sorted({self._time_mapping.get(i, i) for i in s})

    def _update_data_refs(self):
        """Link all loaded time and data columns."""
        for c in self._channels:
            t_col = self._channels[c].time_col
            v_col = self._channels[c].value_col
            if t_col in self._data:
                self._channels[c].time = self._data[t_col]
            if v_col in self._data:
                self._channels[c].values = self._data[v_col]

    def _read_meta_line(self, line, delimiter, expected_fmt):
        """Read a generic metadata line of the header and update channel metadata."""
        key = line.split(delimiter, maxsplit=1)[0]
        values = [sub[:-1].split(delimiter) for sub in line.split(key + delimiter)[1:]]

        # catch wrong formatted SymbolComment
        if (key == "SymbolComment") and (values[-1][-1] == "SymbolComment"):
            values[-1] = values[-1][:-1]
            values = values + [expected_fmt[-1] * [""]]

        values = list(chain(*values))  # flatten

        if not len(self._channels.keys()) == len(values):
            if key == "SymbolComment":
                warn(
                    "Wrong format in 'SymbolComment' header line, skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                return
            else:
                raise ValueError(
                    f"wrong metadata count for {key}"
                    f": {len(self._channels.keys())}, {len(values)}"
                )

        for c, v in zip(self._channels.keys(), values):
            self._channels[c].info[key] = v

    @staticmethod
    def _get_decimal_from_line(line: str, delimiter: str) -> Union[None, str]:
        """Parse the decimal character from a line of numeric values."""
        specials = [
            c for c in line[:-1] if (not (c.isalnum() or c in [delimiter, "-"]))
        ]
        decimal = "".join(set(specials))
        if len(decimal) == 0:
            return None
        elif len(decimal) == 1:
            return decimal
        raise ValueError(f"unexpected {decimal=}")

    def _get_channels(self, channels: list[str] = None) -> dict:
        if isinstance(channels, str):
            channels = [channels]

        if channels is not None:
            return {k: v for k, v in self._channels.items() if k in channels}
        return self._channels

    def _get_time_cols(self, channels: Union[KeysView, list[str]] = None) -> list[int]:
        """Get all associated time columns from a list of channel names."""
        if channels is None:
            channels = self._channels.keys()
        return [v.time_col for v in self._channels.values() if v.name in channels]

    def _get_data_cols(self, channels: Union[KeysView, list[str]] = None) -> list[int]:
        """Get all associated data columns from a list of channel names."""
        if channels is None:
            channels = self._channels.keys()
        return [v.value_col for v in self._channels.values() if v.name in channels]

    def _get_cols(self, channels: list[str] = None) -> list[int]:
        """Get a set of all columns in the data needed for selected channels.

        This contains the columns for time and data.
        """
        d = self._get_channels(channels)

        time_cols = self._get_time_cols(d.keys())
        data_cols = self._get_data_cols(d.keys())

        return sorted(set(time_cols + data_cols))
