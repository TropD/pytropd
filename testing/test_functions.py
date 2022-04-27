import numpy as np
import pytest
from pytropd.functions import (
    find_nearest,
    TropD_Calculate_MaxLat,
    TropD_Calculate_Mon2Season,
    TropD_Calculate_StreamFunction,
    TropD_Calculate_TropopauseHeight,
    TropD_Calculate_ZeroCrossing,
)


EARTH_RADIUS = 6371220.0
GRAV = 9.80616


class TestFindNearest:
    def test_functionality(self):
        data = list(range(4))
        assert data[find_nearest(data, 2.2)] == 2

    def test_nans(self):
        data = list(range(4))
        data[1] = np.nan
        with pytest.warns(RuntimeWarning):
            find_nearest(data, 2)

    def test_skipna(self):
        data = list(range(4))
        data[0] = np.nan
        assert data[find_nearest(data, 2.2, skipna=True)] == 2


class TestMaxLat:
    def test_functionality1D(self, n=6, slice_end=None):
        lats = np.arange(2.5, 92.5, 5)
        data = np.sin(2 * np.radians(lats))[:slice_end]
        assert np.allclose(TropD_Calculate_MaxLat(data, lats, n=n), 45.0)

    def test_nan_interior(self):
        lats = np.arange(2.5, 92.5, 5)
        data = np.sin(2 * np.radians(lats))
        data[4] = np.nan
        with pytest.warns(RuntimeWarning):
            TropD_Calculate_MaxLat(data, lats)

    def test_nan_mid1D(self):
        lats = np.arange(2.5, 92.5, 5)
        data = np.sin(2 * np.radians(lats))
        nandata = np.where((lats > 10.0) & (lats < 80.0), data, np.nan)
        phi = TropD_Calculate_MaxLat(data[2:-2], lats[2:-2])
        nanphi = TropD_Calculate_MaxLat(nandata, lats)
        assert np.allclose(nanphi, phi)

    def test_nan_left1D(self):
        lats = np.arange(2.5, 92.5, 5)
        data = np.sin(2 * np.radians(lats))
        nandata = np.where(lats > 10.0, data, np.nan)
        phi = TropD_Calculate_MaxLat(data[2:], lats[2:])
        nanphi = TropD_Calculate_MaxLat(nandata, lats)
        assert np.allclose(nanphi, phi)

    def test_nan_right1D(self):
        lats = np.arange(2.5, 92.5, 5)
        data = np.sin(2 * np.radians(lats))
        nandata = np.where(lats < 80.0, data, np.nan)
        phi = TropD_Calculate_MaxLat(data[:-2], lats[:-2])
        nanphi = TropD_Calculate_MaxLat(nandata, lats)
        assert np.allclose(nanphi, phi)

    def test_mismatch1D(self):
        with pytest.raises(ValueError):
            self.test_functionality1D(slice_end=-2)

    def test_bad_n(self):
        with pytest.raises(ValueError):
            self.test_functionality1D(n=0)

    def test_functionality2D(self, slice_end=None, transpose=False):
        lats = np.arange(2.5, 92.5, 5)
        data1 = np.sin(2 * np.radians(lats))  # clear max at 45
        data2 = np.sin(2 * np.radians(lats - 15.0))  # shift max to 60
        data3 = np.sin(2 * np.radians(lats + 15.0))  # shift max to 30
        if transpose:
            stack_axis = 1
        else:
            stack_axis = 0
        data = np.stack([data1, data2, data3], axis=stack_axis)[:, :slice_end]

        assert np.allclose(
            TropD_Calculate_MaxLat(data, lats, n=20, axis=not stack_axis),
            np.array([45.0, 60.0, 30.0]),
            atol=0.01,
        )

    def test_mismatch2D(self):
        with pytest.raises(ValueError):
            self.test_functionality2D(slice_end=-2)

    def test_transpose(self):
        self.test_functionality2D(transpose=True)

    def test_bad_axis(self):
        lats = np.arange(2.5, 92.5, 5)
        data1 = np.sin(2 * np.radians(lats))  # clear max at 45
        data2 = np.sin(2 * np.radians(lats - 15.0))  # shift max to 60
        data = np.stack([data1, data2], axis=0)

        with pytest.raises(IndexError):
            TropD_Calculate_MaxLat(data, lats, n=20, axis=3)

    def test_nan2D(self):
        lats = np.arange(2.5, 92.5, 5)
        data1 = np.sin(2 * np.radians(lats))  # clear max at 45
        data2 = np.sin(2 * np.radians(lats - 15.0))  # shift max to 60
        data3 = np.sin(2 * np.radians(lats + 15.0))  # shift max to 30
        data1 = np.where((lats > 10.0) & (lats < 80.0), data1, np.nan)
        data2 = np.where(lats > 5.0, data2, np.nan)
        data3 = np.where(lats < 80.0, data3, np.nan)
        data = np.stack([data1, data2, data3], axis=0)
        nanphi = TropD_Calculate_MaxLat(data, lats)
        phi1 = TropD_Calculate_MaxLat(data1[2:-2], lats[2:-2])
        phi2 = TropD_Calculate_MaxLat(data2[1:], lats[1:])
        phi3 = TropD_Calculate_MaxLat(data3[:-2], lats[:-2])

        assert np.allclose(nanphi, np.stack([phi1, phi2, phi3]))


class TestMon2Season:
    def test_core_functionality(self):
        data = 12 * [1.0, 3.0] + 4 * [7.0, 1.0, 4.0]
        assert (
            TropD_Calculate_Mon2Season(data, range(12)) == np.array([2.0, 2.0, 4.0])
        ).all()

    def test_m(self):
        data = 12 * [1.0, 3.0] + 12 * [
            4.0,
        ]
        with pytest.deprecated_call():
            TropD_Calculate_Mon2Season(data, range(12), m=0)

    def test_bad_axis(self):
        data = 12 * [1.0, 3.0] + 12 * [
            4.0,
        ]
        with pytest.raises(IndexError):
            TropD_Calculate_Mon2Season(data, range(12), axis=3)

    # proof patch works
    def test_indexing_patch1(self, patch=True):
        data = 13 * [1.0, 3.0] + 12 * [
            4.0,
        ]
        result = TropD_Calculate_Mon2Season(
            data, range(12), first_jan_idx=2, patch_indexing=patch
        )
        assert (result == np.array([2.0, 2.0, 4.0])).all()

    def test_indexing_patch2(self, patch=True):
        data = 12 * [1.0, 3.0] + 11 * [
            4.0,
        ]
        result = TropD_Calculate_Mon2Season(data, range(12), patch_indexing=patch)
        assert (result == np.array([2.0, 2.0])).all()

    def test_indexing_patch3(self, patch=True):
        data = 11 * [
            4.0,
        ]
        result = TropD_Calculate_Mon2Season(data, range(12), patch_indexing=patch)
        assert result.size == 0

    def test_indexing_patch4(self, patch=True):
        data = 6 * [1.0, 3.0] + 11 * [
            4.0,
        ]
        result = TropD_Calculate_Mon2Season(data, range(12), patch_indexing=patch)
        if patch:
            ans = np.array([2.0])
        else:
            ans = np.array([2.0, 47.0 / 12.0])
        assert np.allclose(result, ans)

    # bug tests to prove need for patch
    def test_indexing_bug1(self):
        with pytest.raises(ValueError):
            self.test_indexing_patch1(patch=False)

    def test_indexing_bug2(self):
        with pytest.raises(ValueError):
            self.test_indexing_patch2(patch=False)

    def test_indexing_bug3(self):
        with pytest.raises(ValueError):
            self.test_indexing_patch3(patch=False)

    def test_indexing_bug4(self):
        self.test_indexing_patch4(patch=False)


class TestStreamFunction:
    def test_functionality(self, slice_end=None):
        lats = np.arange(0.0, 91.0, 5.0)
        levs = np.linspace(1e5, 1e4, 91)

        def f(y, z):
            return (
                20.0
                * np.cos(np.pi * (z[None, :] - levs[-1]) / (levs[0] - levs[-1]))
                * np.sin(3 * np.pi * y[:, None] / (lats[-1] - lats[0]))
            )

        # antiderivative
        def F(y, z):
            return (
                20.0
                / np.pi
                * (levs[0] - levs[-1])
                * np.sin(np.pi * (1.0 - (z[None, :] - levs[-1]) / (levs[0] - levs[-1])))
                * np.sin(3 * np.pi * y[:, None] / (lats[-1] - lats[0]))
            )

        V = f(lats, levs)[:, :slice_end]
        psi_tropd = TropD_Calculate_StreamFunction(V, lats, levs / 100.0)
        cos_lat = np.cos(np.pi / 180.0 * lats)[:, None]
        psi_exact = (
            (EARTH_RADIUS / GRAV * 2.0 * np.pi)
            * cos_lat
            * (F(lats, levs) - F(lats, levs[:1]))
        )
        assert np.allclose(psi_tropd[:, :-2], psi_exact[:, :-2], rtol=1e-3)

    def test_mismatch(self):
        with pytest.raises(ValueError):
            self.test_functionality(slice_end=-2)


class TestZeroCrossing:
    def test_functionality1D(self, slice_end=None, axis=-1):
        data = np.arange(-22.5, 23.0, 5.0)
        lats = np.linspace(0, 90, data.size)
        assert np.allclose(
            TropD_Calculate_ZeroCrossing(data[:slice_end], lats, axis=axis), 45.0
        )

    def test_bug_exact_zero(self, patch_exact_zero=False):
        data = np.arange(-20.0, 21.0, 4.0)
        lats = np.linspace(0, 90, data.size)
        if patch_exact_zero:
            center = 45.0
        else:
            center = 36.0
        assert np.allclose(
            TropD_Calculate_ZeroCrossing(data, lats, patch_exact_zero=patch_exact_zero),
            center,
        )

    def test_patch_exact_zero(self):
        self.test_bug_exact_zero(patch_exact_zero=True)

    def test_too_small(self):
        data = np.arange(-20, 22.0, 20.0)
        lats = np.linspace(0, 90, data.size)
        with pytest.raises(ValueError):
            TropD_Calculate_ZeroCrossing(data, lats)

    def test_mismatch(self):
        with pytest.raises(ValueError):
            self.test_functionality1D(slice_end=-2)

    def test_bad_axis(self):
        with pytest.raises(IndexError):
            self.test_functionality1D(axis=2)

    def test_lat_uncertainty(self):
        data = np.insert(np.tile(np.arange(-1, 2), 3), 3, [0])
        lats = np.linspace(0, 90, data.size)  # 10-deg spacing
        assert np.isnan(TropD_Calculate_ZeroCrossing(data, lats, lat_uncertainty=22.0))

    def test_no_zc(self):
        data = np.arange(1, 11)
        lats = np.linspace(0, 90, data.size)  # 10-deg spacing
        assert np.isnan(TropD_Calculate_ZeroCrossing(data, lats))

    def test_touch_not_cross(self):
        data = np.array([5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5])
        lats = np.linspace(0, 90, data.size)  # 10-deg spacing
        assert np.isnan(TropD_Calculate_ZeroCrossing(data, lats))

    def test_functionality2D(self, transpose=False):
        data = np.stack([np.arange(-22.5, 23.0, 5.0), np.arange(-12.5, 33.0, 5.0)])
        if transpose:
            data = data.T
            axis = 0
        else:
            axis = -1
        lats = np.linspace(0, 90, data.shape[axis])
        assert np.allclose(
            TropD_Calculate_ZeroCrossing(data, lats, axis=axis), [45.0, 25.0]
        )

    def test_transpose(self):
        self.test_functionality2D(transpose=True)

    def test_nan1D(self):
        data = np.arange(-22.5, 23.0, 5.0)
        data = np.where((-15.0 < data) & (data < 20.0), data, np.nan)
        lats = np.linspace(0, 90, data.size)
        assert np.allclose(TropD_Calculate_ZeroCrossing(data, lats), 45.0)

    def test_nan2D(self):
        data = np.stack([np.arange(-22.5, 23.0, 5.0), np.arange(-12.5, 33.0, 5.0)])
        data = np.where(data > -10.0, data, np.nan)
        lats = np.linspace(0, 90, data.shape[-1])
        assert np.allclose(
            TropD_Calculate_ZeroCrossing(data, lats, axis=-1), [45.0, 25.0]
        )


# TestMaxLat().test_nans()
