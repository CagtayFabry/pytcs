# pytcs

## v0.1.5

### changed

- update license information #49
- add `.github/release.yaml` #49

## v0.1.4

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
