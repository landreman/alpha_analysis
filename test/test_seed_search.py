"""R3.5 bounded seed-discovery regressions against analytic lifted wells."""

import numpy as np

from alpha_analysis.j_connectivity.contour_trace import DirectContourOracle
from alpha_analysis.j_connectivity.seed_search import (
    SeedSearchConfig,
    search_contour_seeds,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField


def _field(m, n, coefficients, iota=0.0):
    cosine = np.asarray(coefficients, dtype=float)
    return SyntheticFourierField(
        1,
        np.asarray(m),
        np.asarray(n),
        cosine,
        np.zeros_like(cosine),
        np.array([iota]),
        np.array([3.0]),
        np.array([0.0]),
    )


def test_seed_search_resumes_without_hiding_unseen_wells():
    """Fixed transverse probes and short lifts cannot imply zero population (§23)."""
    radial = _field([0, 0], [0, 1], [[1.0, 1.0], [0.2, 0.0]])
    transverse = search_contour_seeds(
        DirectContourOracle(radial, 2.0),
        SeedSearchConfig(
            original_locations=((0.3, 0.0), (0.5, 0.0), (0.8, 0.0)),
            supplemental_radii=(0.9,),
            supplemental_angles=1,
            windows=(4, 8),
            max_locations=4,
        ),
    )
    assert not any(a.new_seeds for a in transverse.attempts if a.cohort == "original")
    assert any(a.new_seeds for a in transverse.attempts if a.cohort == "expanded")
    assert all(point.s == 0.9 for point in transverse.seeds)
    assert not transverse.population_empty_proved

    long_lift = _field([0, 1], [0, 0], [[2.0], [1.0]], iota=0.1)
    resumed = search_contour_seeds(
        DirectContourOracle(long_lift, 2.0),
        SeedSearchConfig(
            original_locations=((0.5, 0.0),),
            supplemental_radii=(),
            windows=(4, 8, 16),
            max_locations=1,
        ),
    )
    assert [attempt.periods for attempt in resumed.attempts] == [4, 8, 16]
    assert [attempt.new_seeds for attempt in resumed.attempts[:2]] == [0, 0]
    assert resumed.attempts[0].sample_count < resumed.attempts[-1].sample_count
    assert resumed.seeds
    np.testing.assert_allclose(
        [resumed.seeds[0].zeta_in, resumed.seeds[0].zeta_out],
        [5 * np.pi, 15 * np.pi],
        atol=1e-7,
    )

    no_seed = search_contour_seeds(
        DirectContourOracle(_field([0], [0], [[3.0]]), 2.0),
        SeedSearchConfig(
            original_locations=((0.5, 0.0),),
            supplemental_radii=(),
            windows=(4,),
            max_locations=1,
        ),
    )
    assert no_seed.no_seed_found and no_seed.exhausted
    assert not no_seed.population_empty_proved
