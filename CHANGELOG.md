# pytcs

## unreleased

- add support for reading `StringIO` and `BytesIO` #10

## v0.1.0

initial release with basic functions
- open files and read header metadata using `ScopeFile`
- support different exporting formats (sorted / unsorted time columns, see README)
- read plain text files or gezipped files directly
- load individual channels using `ScopeFile.load()`
- export to `pandas.DataFrame` with `Scope.as_pandas()`
- export to `xarray.DataArray` with `ScopeFile.as_xarray()`
