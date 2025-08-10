"""Test basic io functions."""

from pathlib import Path
from io import BytesIO, StringIO
import tempfile

import numpy as np
import pytest

from pytcs.pytcs import ScopeFile
from pytcs.helpers import get_tc3_dtypes

if np.get_printoptions()["threshold"] == 1000:
    np.set_printoptions(threshold=998)


def file_idfn(path):
    """Get filename."""
    return path.name


# generate filenames to import
files = sorted(Path(".").rglob("**/data/tc3_scope_*.csv*"))
files_pyarrow = [file for file in files if "noOS" in file.name]

# list of files broken for loading:
broken_load = ["Seperator_1", "Seperator_4"]


@pytest.fixture(scope="module", params=files, ids=file_idfn)
def filenames(request):
    return request.param


@pytest.fixture(scope="module", params=files_pyarrow, ids=file_idfn)
def filenames_pyarrow(request):
    return request.param


def _get_as_buffer(file):
    """Convert file into buffer."""
    if file.suffix in [".gz", ".gzip"]:
        with open(file, "rb") as f:
            return BytesIO(f.read())
    else:
        with open(file) as f:
            return StringIO(f.read())


# Testing files in data
class TestScopeFile:
    @staticmethod
    @pytest.mark.parametrize("native_dtypes", [False, True])
    @pytest.mark.parametrize("time_mapping_style", ["full", "reduced"])
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize("use_buffer", [False, True])
    def test_scope_file(
        filenames, time_mapping_style, backend, native_dtypes, use_buffer
    ):
        file = filenames

        if any(sep in str(file) for sep in broken_load):
            pytest.skip("unsupported file format (separators)")

        if use_buffer:
            file = _get_as_buffer(file)

        sf = ScopeFile(file, time_mapping_style=time_mapping_style)
        for c in sf:
            assert sf[c].info

        sf.load(native_dtypes=native_dtypes, backend=backend)

        assert len(sf._get_data_cols()) == len(sf._channels)
        assert len(np.unique(sf._get_time_cols())) + len(sf._get_data_cols()) == len(
            sf._data
        )
        assert all([isinstance(v, np.ndarray) for v in sf._data.values()])

        # monotonic time
        for c in sf:
            assert np.allclose(np.diff(sf[c].time), sf[c].sample_time)

        if native_dtypes:
            tc3 = get_tc3_dtypes()
            for c in sf:
                if c.startswith("var_"):
                    _np_type = tc3[c[4:]][0]
                    assert sf[c]._values.dtype == (_np_type if not sf[c].is_scaled else np.float64)

    @staticmethod
    @pytest.mark.parametrize("native_dtypes", [False, True])
    @pytest.mark.parametrize("use_buffer", [False, True])
    def test_pyarrow_backend(filenames_pyarrow, native_dtypes, use_buffer):
        file = filenames_pyarrow

        if use_buffer:
            file = _get_as_buffer(file)

        sf = ScopeFile(file)
        for c in sf:
            assert sf[c].info

        sf.load(native_dtypes=native_dtypes, backend="pyarrow")

        assert len(sf._get_data_cols()) == len(sf._channels)
        assert len(np.unique(sf._get_time_cols())) + len(sf._get_data_cols()) == len(
            sf._data
        )
        assert all([isinstance(v, np.ndarray) for v in sf._data.values()])

        # monotonic time
        for c in sf:
            assert np.allclose(np.diff(sf[c].time), sf[c].sample_time)

    @staticmethod
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_to_pandas(filenames, backend):
        if any(sep in str(filenames) for sep in broken_load):
            with pytest.raises(ValueError):
                ScopeFile(filenames)
            return None

        sf = ScopeFile(filenames)
        sf.load(backend=backend)
        df = sf.as_pandas()

        assert not df.empty
        assert df.index[0].day == sf.start_time.day

    @staticmethod
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_to_xarray(filenames, backend):
        if any(sep in str(filenames) for sep in broken_load):
            with pytest.raises(ValueError):
                ScopeFile(filenames)
            return None

        sf = ScopeFile(filenames)
        sf.load(backend=backend)
        ds = sf.as_xarray()

        assert ds.variables

        # validate writing to netCDF with attributes
        with tempfile.TemporaryDirectory() as tmpdirname:
            ncfile = tmpdirname + "/test.nc"
            ds.to_netcdf(ncfile)
            assert Path(ncfile).is_file()
