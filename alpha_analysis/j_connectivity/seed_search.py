"""Bounded ordinary-well seed discovery from shared lifted scans (§23 R3.5).

The search samples represented-field lines at deterministic transverse points.
Its fallback locations and resumed windows are finite work, not a proof that
the physical trapped population vanishes when no complete well is found.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contour_trace import ContourPoint, DirectContourOracle
from .forward_catalogue import ForwardLineCatalogue, ForwardScanConfig

ORIGINAL_LOCATIONS = (
    (0.3, 0.0),
    (0.5, 0.0),
    (0.8, 0.0),
    (0.5, float(np.pi / 2)),
    (0.8, float(np.pi / 2)),
)


@dataclass(frozen=True)
class SeedSearchConfig:
    """Finite search work in normalized s, alpha radians and field periods."""

    original_locations: tuple[tuple[float, float], ...] = ORIGINAL_LOCATIONS
    supplemental_radii: tuple[float, ...] = (0.95, 0.99, 0.9, 0.1, 0.02, 0.2)
    supplemental_angles: int = 8
    windows: tuple[int, ...] = (4, 8, 16)
    max_locations: int = 53
    max_seeds: int = 64
    max_seeds_per_location: int = 8

    def __post_init__(self) -> None:
        if (
            not self.windows
            or self.windows[0] < 2
            or any(a >= b for a, b in zip(self.windows[:-1], self.windows[1:]))
            or self.supplemental_angles < 1
            or self.max_locations < 1
            or self.max_seeds < 1
            or self.max_seeds_per_location < 1
        ):
            raise ValueError("invalid finite seed-search work budget")
        if any(
            not 0 <= s <= 1 or not np.isfinite(alpha)
            for s, alpha in self.original_locations
        ):
            raise ValueError("invalid original seed location")
        if any(not 0 <= s <= 1 for s in self.supplemental_radii):
            raise ValueError("invalid supplemental radius")


@dataclass(frozen=True)
class SeedAttempt:
    """One attempted transverse location/window, including its terminal."""

    cohort: str
    s: float
    alpha: float
    periods: int
    status: str
    reason: str | None
    new_seeds: int
    sample_count: int
    root_complete: bool


@dataclass(frozen=True)
class SeedSearchResult:
    """Located lifted wells and all attempted work; no empty-population claim."""

    seeds: tuple[ContourPoint, ...]
    attempts: tuple[SeedAttempt, ...]
    exhausted: bool
    population_empty_proved: bool = False

    @property
    def no_seed_found(self) -> bool:
        return not self.seeds


def search_contour_seeds(
    oracle: DirectContourOracle, config: SeedSearchConfig = SeedSearchConfig()
) -> SeedSearchResult:
    """Find complete selected wells, resuming R1 scans over lifted windows.

    The original five R3 points are replayed at four periods as a fixed
    comparison cohort. Supplemental points are ranked by *sampled* one-period
    support and all are fallback candidates within the work budget; this rank
    never proves emptiness. Each accepted seed retains its physical root lift.
    """
    attempts: list[SeedAttempt] = []
    seeds: list[ContourPoint] = []
    original_for_expansion = []
    processed = 0
    capped = False

    def make_scan(s: float, alpha: float) -> ForwardLineCatalogue:
        sigma = float(np.sign(oracle.field.C(s)))
        if sigma == 0:
            raise ValueError("zero physical scan orientation")
        z0 = -sigma * oracle.period
        return ForwardLineCatalogue(
            oracle.field,
            s,
            alpha + float(oracle.field.iota(s)) * z0,
            z0,
            ForwardScanConfig(max_periods=config.windows[-1]),
        )

    def query_location(cohort, s, alpha, scan, windows):
        nonlocal capped
        seen_lifts: list[float] = []
        accepted = 0
        for periods in windows:
            if (
                len(seeds) >= config.max_seeds
                or accepted >= config.max_seeds_per_location
            ):
                capped = True
                break
            try:
                scan.extend_to(periods)
                result = scan.query(oracle.b)
                candidates = sorted(result.wells, key=lambda w: abs(w.zeta_in))
                fresh = 0
                failures = []
                for well in candidates:
                    if (
                        len(seeds) >= config.max_seeds
                        or accepted >= config.max_seeds_per_location
                    ):
                        capped = True
                        break
                    if any(abs(well.zeta_in - z) < 1e-8 for z in seen_lifts):
                        continue
                    seen_lifts.append(well.zeta_in)
                    try:
                        point = oracle.sample(
                            s,
                            alpha,
                            ContourPoint(s, alpha, well.zeta_in, well.zeta_out, np.nan),
                        )
                    except (ValueError, RuntimeError, ArithmeticError) as error:
                        failures.append(str(error))
                        continue
                    seeds.append(point)
                    fresh += 1
                    accepted += 1
                status = (
                    "FOUND"
                    if fresh and result.root_complete
                    else (
                        "FOUND_WITH_UNRESOLVED_WINDOW"
                        if fresh
                        else (
                            "SEED_SAMPLE_FAILED"
                            if failures
                            else (
                                "NO_COMPLETE_WELL"
                                if result.root_complete
                                else "WINDOW_UNRESOLVED"
                            )
                        )
                    )
                )
                reason = "; ".join(filter(None, (result.reason, *failures))) or None
                attempts.append(
                    SeedAttempt(
                        cohort,
                        s,
                        alpha,
                        periods,
                        status,
                        reason,
                        fresh,
                        scan.sample_count,
                        result.root_complete,
                    )
                )
                if accepted:
                    break
            except (ValueError, ArithmeticError) as error:
                attempts.append(
                    SeedAttempt(
                        cohort,
                        s,
                        alpha,
                        periods,
                        "SCAN_FAILED",
                        str(error),
                        0,
                        scan.sample_count,
                        False,
                    )
                )
                break
        return accepted

    for s, alpha in config.original_locations:
        if processed >= config.max_locations or len(seeds) >= config.max_seeds:
            capped = True
            break
        try:
            scan = make_scan(s, alpha)
        except (ValueError, ArithmeticError) as error:
            attempts.append(
                SeedAttempt(
                    "original", s, alpha, 0, "SCAN_FAILED", str(error), 0, 0, False
                )
            )
        else:
            if not query_location("original", s, alpha, scan, (config.windows[0],)):
                original_for_expansion.append((s, alpha, scan))
        processed += 1

    for s, alpha, scan in original_for_expansion:
        if len(seeds) >= config.max_seeds:
            capped = True
            break
        query_location("expanded_window", s, alpha, scan, config.windows[1:])

    # Priority is diagnostic only. Scan all remaining candidates unless an
    # explicit location/seed budget is reached, preserving the fallback.
    supplemental = []
    for s in config.supplemental_radii:
        for alpha in np.linspace(
            0, 2 * np.pi, config.supplemental_angles, endpoint=False
        ):
            alpha = float(alpha)
            if (s, alpha) in config.original_locations:
                continue
            try:
                scan = make_scan(s, alpha)
                scan.extend_to(1)
                support = bool(
                    np.min(scan.B_samples) < oracle.b < np.max(scan.B_samples)
                )
                supplemental.append((not support, s, alpha, scan))
            except (ValueError, ArithmeticError) as error:
                attempts.append(
                    SeedAttempt(
                        "expanded", s, alpha, 1, "SCAN_FAILED", str(error), 0, 0, False
                    )
                )
    for _, s, alpha, scan in sorted(supplemental, key=lambda item: item[:3]):
        if processed >= config.max_locations or len(seeds) >= config.max_seeds:
            capped = True
            break
        query_location("expanded", s, alpha, scan, config.windows)
        processed += 1
    return SeedSearchResult(
        tuple(seeds),
        tuple(attempts),
        capped
        or processed >= config.max_locations
        or processed == len(config.original_locations) + len(supplemental),
    )
