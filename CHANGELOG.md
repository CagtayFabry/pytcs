# pytcs

## v0.1.2

### added

- add support for reading `StringIO` and `BytesIO` #10

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
