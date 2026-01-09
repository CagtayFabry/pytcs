# pytcs

## v0.2.2 (09.01.2026)

Release `0.2.2` is a maintenance release due to pip install/build errors with netcdf4 1.7.4.

### dependencies

- pin `netcdf4<1.7.4`

## v0.2.1 (02.12.2025)

Release `0.2.1` is a maintenance release to prepare archiving on zenodo.

## v0.2.0 (25.08.2025)

### changed

- add `polars` backend to `ScopeFile.load` #60
- set `polars` as default backend in `ScopeFile.load` #63
- remove `datatable` backend from `ScopeFile.load` #62
- remove `pyarrow` backend from `ScopeFile.load` #65
- assign `BIT` to `uint8` #63
- always read scaled channels as float #63

### dependencies

- add `poalrs>=1.21` #63
- pin `python>=3.11`, `numpy>=2`, `pandas>=2` #62
- remove `pint` dependency #62
- add deprecation warning when using `pandas` backend #65

### general

- update noxfile and pytest github action #62

## v0.1.7 (23.06.2025)

### added

- add `time_mapping_style` parameter to `ScopeFile` #58

## v0.1.6 (14.05.2025)

### fixed

- fix datetime conversion #56

## v0.1.5 (24.04.2025)

### changed

- update license information #49
- add `.github/release.yaml` #49
- update build workflow #51
- determine version at runtime #51

## v0.1.4 (13.03.2025)

### changed

- store `(None)` units as empty strings in `xarray` attributes instead of `(None)` to avoid netCDF incompatibility #48

### dependencies

- drop support for Python 3.8 #25

## v0.1.3

### added

- add experimental support for `pyarrow` pandas backend #18

### fixed

- fixed gzip support when reading from `BytesIO` #18

## v0.1.2

### added

- add support for reading `StringIO` and `BytesIO` #11

### fixed

- explicitly sort index in `ScopeFile.as_pandas` #16

## v0.1.1

initial release with basic functions

- open files and read header metadata using `ScopeFile`
- support different exporting formats (sorted / unsorted time columns, see README)
- read plain text files or gezipped files directly
- load individual channels using `ScopeFile.load()`
- export to `pandas.DataFrame` with `Scope.as_pandas()`
- export to `xarray.DataArray` with `ScopeFile.as_xarray()`
