import numpy as np
import pytest
from temmeta import fitPeaks as fp

# test coordinates
xy1 = np.array([[0, 0],
                [4, 0],
                [0, 3],
                ])

xy2 = np.array([[0, 0],
                [4, 0],
                [0, 3],
                [-3, 0],
                [0, -4],
                ])


def test_calc_nn_network():
    ans1 = fp.calculate_NN_network(xy1)
    exp = np.array([[[0., 0., 4.],
                     [4., 0., 0.],
                     [0., 0., 4.]],
                    [[0., 3., 0.],
                     [0., 0., 3.],
                     [3., 0., 0.]]])
    np.testing.assert_almost_equal(ans1, exp)
    ans2 = fp.calculate_NN_network(xy1, max_r=3)
    exp = np.array([[[0., 0., np.nan],
                     [4., np.nan, np.nan],
                     [0., 0., np.nan]],
                    [[0., 3., np.nan],
                     [0., np.nan, np.nan],
                     [3., 0., np.nan]]])
    np.testing.assert_almost_equal(ans2, exp)
    ans3 = fp.calculate_NN_network(xy1, max_nn=1)
    exp = np.array([[[0.],
                     [4.],
                     [0.]],
                    [[0.],
                     [0.],
                     [3.]]])
    np.testing.assert_almost_equal(ans3, exp)
    ans4 = fp.calculate_NN_network(xy1, max_r=3, max_nn=2)
    exp = np.array([[[0., 0.],
                     [4., np.nan],
                     [0., 0.]],
                    [[0., 3.],
                     [0., np.nan],
                     [3., 0.]]])
    np.testing.assert_almost_equal(ans4, exp)


def test_nn_distance():
    nn = fp.calculate_NN_network(xy1, max_r=3)
    ans = fp.get_NN_distance(nn)
    exp = np.array([[0.,  3., np.nan],
                    [0., np.nan, np.nan],
                    [0.,  3., np.nan]])
    np.testing.assert_almost_equal(ans, exp)


def test_nn_angle():
    nn = fp.calculate_NN_network(xy2)
    ans = fp.get_NN_angle(nn)
    exp = np.array([
        [np.nan,  1.57079633,  3.14159265,          0., -1.57079633],
        [np.nan,  3.14159265,  2.49809154, -2.35619449,  3.14159265],
        [np.nan, -1.57079633, -2.35619449, -0.64350111, -1.57079633],
        [np.nan,          0.,  0.78539816, -0.92729522,          0.],
        [np.nan,  1.57079633,  2.21429744,  0.78539816,  1.57079633]])
    ans = fp.get_NN_angle(nn, units="degrees")
    exp = np.array([
        [np.nan,   90.,  180.,          0.,             -90.],
        [np.nan,  180.,  143.13010235, -135.,           180.],
        [np.nan,  -90., -135.,         -36.86989765,    -90.],
        [np.nan,    0.,   45.,         -53.13010235,    0.],
        [np.nan,   90.,  126.86989765,  45.,            90.]])
    np.testing.assert_almost_equal(ans, exp)
    ans = fp.get_NN_angle(nn, usesense=False, units="degrees")
    exp = np.array([
        [np.nan,   90.,  180.,          0.,             90.],
        [np.nan,  180.,  143.13010235, 135.,           180.],
        [np.nan,  90., 135.,         36.86989765,    90.],
        [np.nan,    0.,   45.,         53.13010235,    0.],
        [np.nan,   90.,  126.86989765,  45.,            90.]])
    np.testing.assert_almost_equal(ans, exp)
    with pytest.raises(ValueError):
        fp.get_NN_angle(nn, vector=[1, 3, 4], units="degrees")
    with pytest.raises(ValueError):
        fp.get_NN_angle(nn, sense=[1, 3, 4], units="degrees")
    with pytest.raises(ValueError):
        fp.get_NN_angle(nn, sense=[1, 3, 4], vector=[1, 1, 1],
                        units="degrees")
    with pytest.raises(ValueError):
        fp.get_NN_angle(nn, vector=[1, 3, 4], units="bullshitunits")
    ans = fp.get_NN_angle(nn, vector=[0, 1], sense=[1, 0], units="degrees")
    exp = np.array([
        [np.nan,  0,  -90,          90.,             180.],
        [np.nan,  -90,  -53.13010235, -135.,           -90.],
        [np.nan,  180, -135.,         126.86989765,    180.],
        [np.nan,    90.,   45.,         143.13010235,    90.],
        [np.nan,   0.,  -36.86989765,  45.,            0.]])
    np.testing.assert_almost_equal(ans, exp)
