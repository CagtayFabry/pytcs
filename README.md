# pytcs: a TwinCAT Scope text export file reader

[![DOI](https://zenodo.org/badge/567243466.svg)](https://doi.org/10.5281/zenodo.17791125)

A Python package for reading exported TwinCAT Scope Files.\
Export your TwinCAT Scope `.svdx` files to `.txt/.csv` and read them into Python.

## quickstart

Open a file and create a `pandas.DataFrame`:

```python
from pytcs import ScopeFile

sf = ScopeFile("example.csv")  # open file and read metadata
df = sf.as_pandas()  # convert to pandas DataFrame
```

## user guide

### installation

Install using `pip/uv` or `conda/mamba`:

```
pip install pytcs
```

```
conda install pytcs
```

### loading data

To get started, open a file using `pytcs.ScopeFile`:

```python
from pytcs import ScopeFile

sf = ScopeFile("example.csv")
sf
# > <TwinCAT Scope File at 0x2a157ca9310>
# > name:    example
# > runtime: 0:00:00.999000
# > start:   2022-05-02T13:56:24.376000+00:00
# >
# > Channels:
# >   *var_REAL64: 1.0 ms [None]
# >   *var_UINT64: 1.0 ms [None]
# >   *var_UINT32: 1.0 ms [None]
# >   *var_UINT16: 1.0 ms [None]
# >   *func_units_scaled: 1.0 ms [dV]
```

You can see the list of channels contained in the file together with the sample time and the unit.
When creating a `ScopeFile` instance only the metadata about the channels is read from the file header.
The actual channel data is not loaded, indicated by the `*` in front of the channels.

You can load all or only a list of channels by using `ScopeFile.load()`

```python
sf.load()
sf
# > <TwinCAT Scope File at 0x2a157ca9310>
# > name:    example
# > runtime: 0:00:00.999000
# > start:   2022-05-02T13:56:24.376000+00:00
# >
# > Channels:
# >   var_REAL64: 1.0 ms [None]
# >   var_UINT64: 1.0 ms [None]
# >   var_UINT32: 1.0 ms [None]
# >   var_UINT16: 1.0 ms [None]
# >   func_units_scaled: 1.0 ms [dV]
```

### accessing individual channels

Individual channels can be accessed by their name:

```python
sf["func_units_scaled"]
# > ScopeChannel(name='func_units_scaled',
# >   time=array([  0.,   1.,   2., ..., 997., 998., 999.]),
# >   values=array([   10.,     0.,   -10., ..., -9960., -9970., -9980.]),
# >   sample_time=1.0, time_offset=0.0, units='dV')
```

### CSV backends

The default implementation of `pytcs` uses `polars.read_csv` for parsing CSV files.
The `polars` backend is the new default since `v0.2.0` and aims to provide support for the various formatting options provided by the TwinCAT Scope export tool with good file reading performance.

In case of compatibility issues, the old default `pandas` backend can be used calling `ScopeFile.load(..., backend="pandas")`.
The `pandas` backend will be deprecated in a future release.

### exporting to pandas and xarray

To work with the data, convert them to a pandas or xarray object. Channels will automatically be loaded form the file when exporting to other formats.
You can select individual channels to export.

```python
sf.as_pandas(channels=["var_REAL64", "var_UINT16"])
# >                          var_REAL64  var_UINT16
# > time
# > 2022-05-02 13:56:24.376         0.0         0.0
# > 2022-05-02 13:56:24.377         1.0         1.0
# > 2022-05-02 13:56:24.378         2.0         2.0
# > 2022-05-02 13:56:24.379         3.0         3.0
# > 2022-05-02 13:56:24.380         4.0         4.0
# > ...                             ...         ...
# > 2022-05-02 13:56:25.371       995.0       995.0
# > 2022-05-02 13:56:25.372       996.0       996.0
# > 2022-05-02 13:56:25.373       997.0       997.0
# > 2022-05-02 13:56:25.374       998.0       998.0
# > 2022-05-02 13:56:25.375       999.0       999.0
# >
# > [1000 rows x 2 columns]
```

Exporting to an `xarray.Dataset` will preserve the metadata as attributes.

```python
sf.as_xarray(channels=["var_REAL64", "var_UINT16"])
# > <xarray.Dataset>
# > Dimensions:     (time: 1000)
# > Coordinates:
# >   * time        (time) datetime64[ns] 2022-05-02T13:56:24.376000 ... 2022-05-...
# > Data variables:
# >     var_REAL64  (time) float64 0.0 1.0 2.0 3.0 4.0 ... 996.0 997.0 998.0 999.0
# >     var_UINT16  (time) float64 0.0 1.0 2.0 3.0 4.0 ... 996.0 997.0 998.0 999.0
# > Attributes:
# >     ScopeName:   tc3_scope_3_4_3145_3
# >     File:        C:\Python\weldx-dev\pytcs\tests\data\tc3_scope_3_4_3145_3-Co...
# >     StartTime:   132959733843760000
# >     EndTime:     132959733853750000
# >     start_time:  2022-05-02T13:56:24.376000+00:00
# >     run_time:    0:00:00.999000
```

### dtype support

By default, all data will be read as `np.float64`.
When importing data with `ScopeFile.load` using the option `native_dtypes=True`, imported data will be converted to their native dtypes.

| TwinCAT Scope | numpy        | IEC61131-3 |
| ------------- | ------------ | ---------- |
| BIT           | `np.bool_`   | BOOL       |
| INT8          | `np.int8`    | SINT       |
| INT16         | `np.int16`   | INT        |
| INT32         | `np.int32`   | DINT       |
| INT64         | `np.int64`   | LINT       |
| UINT8         | `np.uint8`   | USINT      |
| UINT16        | `np.uint16`  | UINT       |
| UINT32        | `np.uint32`  | UDINT      |
| UINT64        | `np.uint64`  | ULINT      |
| REAL32        | `np.float32` | REAL       |
| REAL64        | `np.float64` | LREAL      |

### export options support

The following table lists the compatible ✅ and currently uncompatible ❌ options of the ScopeExporter:

| file and value formats              |     |
| ----------------------------------- | --- |
| **ScaleValues**                     |     |
| true                                | ✅  |
| false                               | ❌  |
| **DecimalMark**                     |     |
| `.`                                 | ✅  |
| `,`                                 | ✅  |
| **Seperator**                       |     |
| Tab                                 | ✅  |
| Blank (space)                       | ❌  |
| Colon                               | ✅  |
| Semicolon                           | ✅  |
| Comma                               | ❌  |
| **ExcludeDoubleTimestamp**          |     |
| true                                | ✅  |
| false                               | ✅  |
| **SortChannels**                    |     |
| true                                | ✅  |
| false                               | ✅  |
| **FullTimeStamp**                   |     |
| true                                | ❌  |
| false                               | ✅  |
| **AdditionalEmptyLine**             |     |
| true                                | ❌  |
| false                               | ✅  |
| **ContainEOF**                      |     |
| true                                | ✅  |
| false                               | ✅  |
| **HeaderKonfiguration**             |     |
| Full Header                         | ✅  |
| **ArraySeperator**                  |     |
| Tab                                 | ✅  |
| **AdditionalArraySeperator**        |     |
| true                                | ❌  |
| false                               | ✅  |
| **IncludeTriggerInfos**             |     |
| true                                | ❌  |
| false                               | ✅  |
| **IncludeMarkerTables**             |     |
| None                                | ✅  |
| **MarkerTableOnlyIncludedChannels** |     |
| true                                | ❌  |
| false                               | ✅  |
| **MarkerTableOnlyIncludedMarker**   |     |
| true                                | ❌  |
| false                               | ✅  |
