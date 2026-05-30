import numpy as np

from nfnet.noise import MutableUniformNoise
from nfnet.callbacks import CountController, TrainingState


def test_mutable_uniform_sigma_settable_and_callable():
    rng = np.random.default_rng(0)
    nz = MutableUniformNoise(sigma=0.2)
    assert not hasattr(nz, "observe")          # plain injector, not activity-aware
    draws = np.array([nz(0, 5, rng) for _ in range(4000)])
    assert abs(draws.std() - 0.2) < 0.03
    nz.sigma = 0.5
    draws2 = np.array([nz(0, 5, rng) for _ in range(4000)])
    assert draws2.std() > draws.std()


def _state(n_active, n_total, n_inputs=8, step=100):
    W = np.zeros((n_total, n_inputs))
    W[:n_active] = 1.0                          # active rows have norm > floor
    return TrainingState(step=step, total_steps=1000, eta=0.01, weights=W,
                         outputs=np.zeros(n_total))


def test_controller_raises_sigma_when_too_many_active():
    nz = MutableUniformNoise(sigma=0.1)
    ctl = CountController(nz, target=4, every=100, factor=1.5)
    s0 = nz.sigma
    ctl.on_step(_state(n_active=8, n_total=12, step=100))
    assert nz.sigma > s0


def test_controller_lowers_sigma_when_too_few_active():
    nz = MutableUniformNoise(sigma=0.1)
    ctl = CountController(nz, target=8, every=100, factor=1.5)
    s0 = nz.sigma
    ctl.on_step(_state(n_active=4, n_total=12, step=100))
    assert nz.sigma < s0


def test_controller_acts_only_on_interval():
    nz = MutableUniformNoise(sigma=0.1)
    ctl = CountController(nz, target=4, every=100, factor=1.5)
    s0 = nz.sigma
    ctl.on_step(_state(n_active=8, n_total=12, step=37))    # not a multiple of `every`
    assert nz.sigma == s0


def test_controller_respects_sigma_bounds():
    nz = MutableUniformNoise(sigma=1.9)
    ctl = CountController(nz, target=1, every=1, factor=2.0, sigma_max=2.0)
    ctl.on_step(_state(n_active=12, n_total=12, step=1))    # would exceed max
    assert nz.sigma <= 2.0
