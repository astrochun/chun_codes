import astropy.units as u
import numpy as np

from chun_codes import cardelli


def test_cardelli():

    line_Ha = 6562.8
    lambda0 = line_Ha * u.Angstrom
    result1 = cardelli.cardelli(lambda0)
    assert isinstance(result1, float)

    result2 = cardelli.cardelli([line_Ha] * u.Angstrom)
    assert isinstance(result2, float)

    lambda0 = [500, 900, 1100, 1500, 2500, 5500, 15000] * u.Angstrom
    result3 = cardelli.cardelli(lambda0)
    assert isinstance(result3, (np.ndarray, np.generic))
    assert len(result3) == len(lambda0)
