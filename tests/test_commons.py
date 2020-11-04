import numpy as np

import chun_codes


def test_systime():
    assert type(chun_codes.systime()) == str


def test_intersect():
    a = [1, 3, 4]
    b = [3, 4, 5, 6]
    commons = chun_codes.intersect(a, b)
    assert len(commons) == 2
    assert np.array_equal(commons, np.array([3, 4]))


def test_random_pdf():
    x = [1, 3, 4]
    dx = [0.1, 0.15, 0.21]
    n_iter = 5000
    pdf = chun_codes.random_pdf(x, dx, n_iter=n_iter)
    assert isinstance(pdf, (np.ndarray, np.generic))
    assert pdf.shape[0] == len(x)
    assert pdf.shape[1] == n_iter


def test_compute_onesig_pdf():
    x = [1, 3, 4]
    dx = [0.1, 0.15, 0.21]
    n_iter = 5000
    pdf = chun_codes.random_pdf(x, dx, n_iter=n_iter)

    err, peak = chun_codes.compute_onesig_pdf(pdf, x, usepeak=True)
    assert isinstance(err, (np.ndarray, np.generic))
    assert isinstance(peak, (np.ndarray, np.generic))
    assert err.shape[0] == len(x)
    assert err.shape[1] == 2
    assert len(peak) == len(x)


def test_quad_low_high_err():
    x = [1, 3, 4]
    dx = [0.1, 0.15, 0.21]
    n_iter = 5000
    pdf = chun_codes.random_pdf(x, dx, n_iter=n_iter)

    err, peak = chun_codes.compute_onesig_pdf(pdf, x, usepeak=True)
    err_added = chun_codes.quad_low_high_err(err)
    assert len(err_added) == len(x)

    # Check hi case:
    err_hi = chun_codes.quad_low_high_err(err_added, hi=np.array(dx))
    assert len(err_hi) == len(x)


def test_rem_dup():
    values = [3, 4, 5, 1, 4]
    values0 = chun_codes.rem_dup(values)

    assert isinstance(values0, (np.ndarray, np.generic))
    assert len(values0) == len(set(values))
