import numpy as np

from nfnet.nonlinearities import soft_shrink, soft_threshold


def test_soft_shrink_is_odd_symmetric():
    f = soft_shrink(tau=1.0, lam=4.0)
    a = np.array([-3.0, -1.0, -0.2, 0.2, 1.0, 3.0])
    np.testing.assert_allclose(f(-a), -f(a), rtol=1e-6, atol=1e-9)


def test_soft_shrink_zero_at_origin():
    f = soft_shrink(tau=1.0, lam=4.0)
    assert abs(float(f(np.array([0.0]))[0])) < 1e-9


def test_soft_shrink_preserves_sign_and_squashes_small():
    f = soft_shrink(tau=1.0, lam=4.0)
    vals = f(np.array([-3.0, -0.1, 0.1, 3.0]))
    assert vals[0] < 0 and vals[3] > 0            # sign preserved for large |a|
    assert abs(vals[1]) < 0.1 and abs(vals[2]) < 0.1  # small |a| squashed toward 0
    # large positive activation approaches a - tau (softplus saturates):
    big = float(f(np.array([5.0]))[0])
    assert abs(big - (5.0 - 1.0)) < 0.2


def test_soft_shrink_has_name():
    f = soft_shrink(tau=1.0, lam=4.0)
    assert "soft_shrink" in f.name
