"""Test basic io functions."""
from pathlib import Path
from io import BytesIO, StringIO

import numpy as np
import pytest

from pytcs.pytcs import ScopeFile

if np.get_printoptions()["threshold"] == 1000:
    np.set_printoptions(threshold=998)


def file_idfn(path):
    """Get filename."""
    return path.name


# generate filenames to import
files = list(Path(".").rglob("**/data/tc3_scope_*.csv"))
# files = list(Path(".").rglob("**/data/vs_eng*"))

# list of files broken for loading:
broken_load = ["Seperator_1", "Seperator_4"]


@pytest.fixture(scope="module", params=files, ids=file_idfn)
def filenames(request):
    return request.param


# Testing files in data
class TestScopeFile:
    @staticmethod
    @pytest.mark.parametrize("native_dtypes", [False, True])
    @pytest.mark.parametrize("backend", ["pandas"])
    @pytest.mark.parametrize("use_buffer", [False, True])
    def test_scope_file(filenames, backend, native_dtypes, use_buffer):
        if native_dtypes & (backend == "datatable"):
            pytest.skip("unsupported configuration")

        if any(sep in str(filenames) for sep in broken_load):
            with pytest.raises(ValueError):
                ScopeFile(filenames)
            return None

        if use_buffer:
            if filenames.suffix in [".gz", ".gzip"]:
                with open(filenames, "rb") as f:
                    filenames = BytesIO(f.read())
            else:
                with open(filenames, "rt") as f:
                    filenames = StringIO(f.read())

        sf = ScopeFile(filenames)
        for c in sf:
            assert sf[c].info

        sf.load(native_dtypes=native_dtypes, backend=backend)

        assert len(sf._get_data_cols()) == len(sf._channels)
        assert len(np.unique(sf._get_time_cols())) + len(sf._get_data_cols()) == len(
            sf._data
        )
        assert all([type(v) == np.ndarray for v in sf._data.values()])

        # monotonic time
        for c in sf:
            assert np.allclose(np.diff(sf[c].time), sf[c].sample_time)

    @staticmethod
    @pytest.mark.parametrize("backend", ["pandas"])
    def test_to_pandas(filenames, backend):
        if any(sep in str(filenames) for sep in broken_load):
            with pytest.raises(ValueError):
                ScopeFile(filenames)
            return None

        sf = ScopeFile(filenames)
        sf.load(backend=backend)
        df = sf.as_pandas()

        assert not df.empty

    @staticmethod
    @pytest.mark.parametrize("backend", ["pandas"])
    def test_to_xarray(filenames, backend):
        if any(sep in str(filenames) for sep in broken_load):
            with pytest.raises(ValueError):
                ScopeFile(filenames)
            return None

        sf = ScopeFile(filenames)
        sf.load(backend=backend)
        ds = sf.as_xarray()

        assert ds.variables
