"""Local, root-labelled well charts and marginal ports (DESIGN.md §§8, 10).

The transverse certificate below is scoped to a finite represented-field window.
Unsupported field representations and ambiguous cells remain unknown. In
particular, sampled root counts are never promoted to cell multiplicities.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

from ..boozer_field import BoozerField
from .field import BoozerFieldLike
from .forward_catalogue import ForwardLineCatalogue, ForwardScanConfig
from .synthetic_fields import SyntheticFourierField


@dataclass(frozen=True)
class AtlasConfig:
    """Uniform transverse cells in normalized flux s and alpha radians (§8.2)."""

    n_s: int = 3
    n_alpha: int = 8
    periods: int = 2
    cell_subdivisions: int = 8
    compute_actions: bool = False

    def __post_init__(self) -> None:
        if self.n_s < 2 or self.n_alpha < 2 or self.periods < 1:
            raise ValueError("atlas grid and periods must have at least two, two, one")
        if self.cell_subdivisions < 0:
            raise ValueError("cell_subdivisions must be nonnegative")


@dataclass(frozen=True)
class AtlasWell:
    """One owned physical incoming root and its first outgoing root (§8.1).

    ``branch_id`` is a local root ordinal, meaningful only inside certified
    neighboring cells. zeta lifts are radians; A is a length or NaN if omitted.
    """

    branch_id: int
    s: float
    alpha: float
    zeta_in: float
    zeta_out: float
    action_length: float = np.nan
    action_status: str = "not_computed"
    action_reason: str | None = None


@dataclass(frozen=True)
class AtlasSample:
    """Sampled chart state; uncensored ownership is not cell coverage (§8.2)."""

    s: float
    alpha: float
    wells: tuple[AtlasWell, ...]
    reason: str | None


@dataclass(frozen=True)
class AtlasCell:
    """Transverse area and finite-window multiplicity scope (§8.3–8.4)."""

    s_interval: tuple[float, float]
    alpha_interval: tuple[float, float]
    area: float
    multiplicity_lower: int
    multiplicity_upper: int | None
    unknown_reason: str | None
    matched_local_branches: tuple[int, ...] = ()


@dataclass(frozen=True)
class BranchAtlas:
    """Owned sampled wells plus certified/unknown chart cells for one b (§8)."""

    field: BoozerFieldLike
    b: float
    samples: tuple[AtlasSample, ...]
    cells: tuple[AtlasCell, ...]
    config: AtlasConfig

    @property
    def owned_wells(self) -> tuple[AtlasWell, ...]:
        """Sampled owned wells, collapsing axis labels only for axis-regular fields.

        BoozerField uses the §7.3 core continuation. A synthetic polynomial
        field has an equivalent axis line only when all m!=0 modes vanish at
        s=0. Unknown field types retain every axis sample conservatively.
        """
        if isinstance(self.field, BoozerField):
            collapse_axis = True
        elif isinstance(self.field, SyntheticFourierField):
            transverse = self.field.m != 0
            collapse_axis = not np.any(
                self.field.cosine_coefficients[transverse, 0]
            ) and not np.any(self.field.sine_coefficients[transverse, 0])
        else:
            collapse_axis = False
        return tuple(
            well
            for sample in self.samples
            if not collapse_axis or sample.s != 0 or sample.alpha == 0
            for well in sample.wells
        )

    @property
    def unknown_area(self) -> float:
        return float(sum(c.area for c in self.cells if c.multiplicity_upper is None))

    @property
    def known_owned_area(self) -> float:
        """Unweighted |ds d alpha| times certified multiplicity, not K weight."""
        return float(
            sum(
                c.area * c.multiplicity_lower
                for c in self.cells
                if c.multiplicity_upper is not None
            )
        )

    @property
    def multiplicity_area_upper(self) -> float | None:
        """Upper integrated count×|ds dα|, or unknown; not a K-weight bound."""
        if self.unknown_area:
            return None
        return self.known_owned_area


@dataclass(frozen=True)
class HeightSample:
    """Local nondegenerate extremum H and ∇H, in B units (§10.1)."""

    s: float
    alpha: float
    zeta: float
    value: float
    curvature: float
    gradient_s: float
    gradient_alpha: float


@dataclass(frozen=True)
class AtlasPort:
    """Pointwise limiting well at common (s, alpha), unbound to an atlas branch (§10.2)."""

    role: str
    event_parameter: tuple[float, float]
    zeta_in: float
    zeta_out: float
    action_length: float
    error_estimate: float


@dataclass(frozen=True)
class AtlasTransition:
    """Pointwise three-port relation or enclosed multiway event (§10.2–10.3).

    A generic local event is not a certified port curve or atlas branch binding.
    """

    status: str
    parameter: tuple[float, float]
    marginal_zeta: tuple[float, ...]
    ports: tuple[AtlasPort, ...]
    reason: str | None = None


def seam_image(
    alpha: float, s: float, field: BoozerFieldLike, periods: int = 1
) -> float:
    """Map a zeta=L lift to zeta=0: alpha -> alpha+iota(s)L (§8.2)."""
    return float(alpha + periods * float(field.iota(s)) * 2 * np.pi / field.nfp)


def identify_lifted_overlap(
    field: BoozerFieldLike,
    first: AtlasWell,
    second: AtlasWell,
    period_shift: int,
    atol: float = 1e-9,
) -> bool:
    """Identify the same entry/first-exit root under an explicit seam lift (§8.2).

    A specified integer window shift and both continued endpoints are required;
    similar physical coordinates or equal local ordinals alone do not merge wells.
    The two branch records are in radian zeta lifts and normalized flux s.
    """
    if not isinstance(period_shift, int) or not np.isfinite(atol) or atol <= 0:
        raise ValueError("period_shift must be integer and atol positive")
    if abs(first.s - second.s) > atol:
        return False
    L = 2 * np.pi / field.nfp
    if (
        abs(second.zeta_in - (first.zeta_in - period_shift * L)) > atol
        or abs(second.zeta_out - (first.zeta_out - period_shift * L)) > atol
    ):
        return False
    expected_alpha = seam_image(first.alpha, first.s, field, period_shift)
    return abs(np.angle(np.exp(1j * (second.alpha - expected_alpha)))) <= atol


def _line(field: BoozerFieldLike, s: float, alpha: float, periods: int):
    L = 2 * np.pi / field.nfp
    sigma = np.sign(float(field.C(s)))
    if sigma == 0:
        raise ValueError("zero G+iota I has no physical scan orientation")
    z0 = 0.0 if sigma > 0 else L
    theta0 = alpha + float(field.iota(s)) * z0
    scan = ForwardLineCatalogue(
        field, s, theta0, z0, ForwardScanConfig(max_periods=periods)
    )
    scan.extend_to(periods)
    return scan


def _owned(scan: ForwardLineCatalogue, b: float, compute_actions: bool):
    result = scan.query(b)
    L = scan.period
    wells = [w for w in result.wells if 0 <= w.u_in < L]
    actions = [np.nan] * len(wells)
    action_status = ["not_computed"] * len(wells)
    action_reasons = [None] * len(wells)
    if compute_actions and wells:
        from .forward_catalogue import batched_bounce_integrals

        integrals = batched_bounce_integrals(scan, wells)
        actions = [item.A for item in integrals]
        action_status = [item.status.name for item in integrals]
        action_reasons = [item.reason for item in integrals]
    owned = tuple(
        AtlasWell(
            i,
            scan.s,
            scan.theta0 - scan.iota * scan.zeta0,
            well.zeta_in,
            well.zeta_out,
            float(actions[i]),
            action_status[i],
            action_reasons[i],
        )
        for i, well in enumerate(wells)
    )
    return owned, result.reason


def _transverse_bounds(
    field: BoozerFieldLike, s0: float, s1: float, alpha_width: float, z_abs: float
):
    """Global polynomial/Fourier derivative envelopes over one q box.

    This is a represented-field bound, not a measurement of interpolation or
    coefficient error. Unsupported radial representations stay unknown.
    """
    if isinstance(field, SyntheticFourierField):
        cosine, sine = field.cosine_coefficients, field.sine_coefficients
        powers = np.arange(cosine.shape[1], dtype=float)
        coeff_abs = np.sum(np.abs(cosine) + np.abs(sine), axis=1)
        derivative_abs = np.sum(
            (np.abs(cosine[:, 1:]) + np.abs(sine[:, 1:])) * powers[1:], axis=1
        )
        iota_abs = np.sum(np.abs(field.iota_coefficients))
        shear_abs = np.sum(
            np.abs(field.iota_coefficients[1:])
            * np.arange(1, len(field.iota_coefficients))
        )
        m, n = field.m, field.n
    elif all(
        hasattr(field, name)
        for name in ("_bmnc_spline", "_bmns_spline", "_iota_spline", "xm", "xn")
    ):
        cosine = _coefficient_spline_bounds(field, field._bmnc_spline, s0, s1)
        sine = _coefficient_spline_bounds(field, field._bmns_spline, s0, s1)
        if cosine is None or sine is None:
            return None
        coeff_abs = cosine[0] + sine[0]
        derivative_abs = cosine[1] + sine[1]
        iota_abs, shear_abs = _spline_abs_bounds(field._iota_spline, s0, s1)
        m, n = np.asarray(field.xm), np.asarray(field.xn)
    else:
        return None
    freq_abs = np.abs(m) * iota_abs + np.abs(n)
    radius = (s1 - s0) / 2
    angle = alpha_width / 2
    variation = np.sum(
        derivative_abs * radius
        + coeff_abs * np.abs(m) * (angle + shear_abs * radius * z_abs)
    )
    derivative_variation = np.sum(
        derivative_abs * radius * freq_abs
        + coeff_abs * np.abs(m) * shear_abs * radius
        + coeff_abs * freq_abs * np.abs(m) * (angle + shear_abs * radius * z_abs)
    )
    curvature = np.sum(coeff_abs * freq_abs**2)
    inflation = 1 + 128 * np.finfo(float).eps * len(coeff_abs)
    return (
        float(variation * inflation),
        float(derivative_variation * inflation),
        float(curvature * inflation),
    )


def _spline_abs_bounds(spline, s0, s1):
    """Conservative cubic and first-derivative bounds on a radial interval.

    CubicSpline.c is local monomial data. The triangle inequality covers
    knot segments and the library's endpoint extrapolation exactly as a model.
    """
    cuts = [s0] + [float(x) for x in spline.x if s0 < x < s1] + [s1]
    values = []
    derivatives = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        mid = (left + right) / 2
        index = int(
            np.clip(
                np.searchsorted(spline.x, mid, side="right") - 1, 0, len(spline.x) - 2
            )
        )
        t = max(abs(left - spline.x[index]), abs(right - spline.x[index]))
        c = np.abs(spline.c[:, index])
        values.append(c[0] * t**3 + c[1] * t**2 + c[2] * t + c[3])
        derivatives.append(3 * c[0] * t**2 + 2 * c[1] * t + c[2])
    return np.max(values, axis=0), np.max(derivatives, axis=0)


def _coefficient_spline_bounds(field, spline, s0, s1):
    core = field._coefficient_s0
    if core is None or s0 >= core:
        return _spline_abs_bounds(spline, s0, s1)
    m = np.abs(np.asarray(field.xm, dtype=float))
    axis_hi = min(s1, core)
    value, derivative = _spline_abs_bounds(spline, s0, axis_hi)
    nonzero = m > 0
    p = m[nonzero] / 2
    anchor = np.abs(np.asarray(spline(core))[nonzero])
    value[nonzero] = anchor * (axis_hi / core) ** p
    if s0 == 0 and np.any(p < 1):
        return None
    derivative[nonzero] = (
        anchor
        * p
        / core
        * np.where(p < 1, (s0 / core) ** (p - 1), (axis_hi / core) ** (p - 1))
    )
    if s1 > core:
        outer_value, outer_derivative = _spline_abs_bounds(spline, core, s1)
        value = np.maximum(value, outer_value)
        derivative = np.maximum(derivative, outer_derivative)
    return value, derivative


def _orientation_certified(field, s0, s1):
    s = (s0 + s1) / 2
    radius = (s1 - s0) / 2
    if isinstance(field, SyntheticFourierField):
        C_coefficients = np.polynomial.polynomial.polyadd(
            field.G_coefficients,
            np.polynomial.polynomial.polymul(
                field.iota_coefficients, field.I_coefficients
            ),
        )
        bound = np.sum(np.abs(C_coefficients[1:]) * np.arange(1, len(C_coefficients)))
    elif all(
        hasattr(field, name) for name in ("_G_spline", "_I_spline", "_iota_spline")
    ):
        G, dG = _spline_abs_bounds(field._G_spline, s0, s1)
        I, dI = _spline_abs_bounds(field._I_spline, s0, s1)
        iota, diota = _spline_abs_bounds(field._iota_spline, s0, s1)
        bound = dG + diota * I + iota * dI
    else:
        return False
    return abs(float(field.C(s))) > float(bound) * radius


def _certified_roots(
    scan: ForwardLineCatalogue,
    b: float,
    q_bounds: tuple[float, float, float, float],
    subdivisions: int,
):
    """Interval-exclude or prove one monotone crossing for every q in a cell."""
    s0, s1, a0, a1 = q_bounds
    z_abs = max(
        abs(scan.zeta0),
        abs(scan.zeta0 + scan.sigma * scan.period * scan.scanned_periods),
    )
    envelopes = _transverse_bounds(scan.field, s0, s1, a1 - a0, z_abs)
    if envelopes is None:
        return None, "no validated transverse field envelope"
    vB, vD, M = envelopes
    rounding = (
        128 * np.finfo(float).eps * max(1.0, abs(b), np.max(np.abs(scan.B_samples)))
    )
    if vB > 0 and np.any(np.abs(scan.B_samples - b) <= vB + rounding):
        return (
            None,
            f"transverse B envelope (vB={vB:.3g} B units) overlaps b at a scan node; "
            "narrow the s/alpha cell",
        )
    roots = []

    def visit(l, r, Bl, Br, Dl, Dr, depth):
        h = r - l
        padding = vB + M * h * h / 8 + rounding
        if max(Bl, Br) + padding < b or min(Bl, Br) - padding > b:
            return True
        low_d = min(Dl, Dr) - vD - M * h
        high_d = max(Dl, Dr) + vD + M * h
        if low_d > 0 or high_d < 0:
            left_sign = np.sign(Bl - b) if abs(Bl - b) > vB + rounding else 0
            right_sign = np.sign(Br - b) if abs(Br - b) > vB + rounding else 0
            if vB == 0 and left_sign == 0 and right_sign:
                roots.append((l, l, 1 if low_d > 0 else -1))
                return True
            if vB == 0 and right_sign == 0 and left_sign:
                roots.append((r, r, 1 if low_d > 0 else -1))
                return True
            if left_sign and right_sign and left_sign != right_sign:
                roots.append((l, r, 1 if low_d > 0 else -1))
                return True
            if left_sign and right_sign and left_sign == right_sign:
                return True
        if depth >= subdivisions:
            return False
        mid = (l + r) / 2
        Bm, Dm = float(scan._B(mid)), float(scan._D(mid))
        return visit(l, mid, Bl, Bm, Dl, Dm, depth + 1) and visit(
            mid, r, Bm, Br, Dm, Dr, depth + 1
        )

    for i in range(len(scan.u) - 1):
        if not visit(
            scan.u[i],
            scan.u[i + 1],
            scan.B_samples[i],
            scan.B_samples[i + 1],
            scan.D_samples[i],
            scan.D_samples[i + 1],
            0,
        ):
            return None, "possible hidden barrier or transverse root change"
    roots.sort()
    unique = []
    for root in roots:
        if (
            unique
            and abs(root[0] - unique[-1][0]) < 1e-12
            and root[0] == root[1] == unique[-1][1]
        ):
            if root[2] != unique[-1][2]:
                return None, "inconsistent endpoint root orientation"
            continue
        unique.append(root)
    return unique, None


def _certify_cell(field, b, bounds, periods, subdivisions):
    s0, s1, a0, a1 = bounds
    s, alpha = (s0 + s1) / 2, (a0 + a1) / 2
    if not _orientation_certified(field, s0, s1):
        return None, "physical scan orientation may change inside cell"
    try:
        scan = _line(field, s, alpha, periods)
        roots, reason = _certified_roots(scan, b, bounds, subdivisions)
    except (ValueError, ArithmeticError) as error:
        return None, str(error)
    if roots is None:
        return None, reason
    # A root interval touching the scan-core boundary cannot be assigned
    # uniquely to this chart without another lift.
    L = scan.period
    if abs(float(scan._B(0)) - b) <= 1e-9 or abs(float(scan._B(L)) - b) <= 1e-9:
        return None, "incoming root may cross chart window boundary"
    if roots and any(l < L < r for l, r, _ in roots):
        return None, "incoming root may cross chart window boundary"
    count = 0
    waiting = False
    for left, right, sign in roots:
        if sign < 0:
            if waiting:
                return None, "root ordering cannot establish first return"
            waiting = left < L
        elif waiting:
            count += 1
            waiting = False
    if waiting:
        return None, "first outgoing root beyond certified window"
    return count, None


def classify_cell(
    field: BoozerFieldLike,
    b: float,
    s_interval: tuple[float, float],
    alpha_interval: tuple[float, float],
    periods: int = 2,
    subdivisions: int = 8,
) -> AtlasCell:
    """Certify a regular root pattern or enclose unknown multiplicity (§8.4).

    Bounds cover the represented Fourier/polynomial or Fourier/spline field and
    a finite scan window, not the physical field's representation error.
    Unknown cells have no exact local population assignment.
    """
    s0, s1 = map(float, s_interval)
    a0, a1 = map(float, alpha_interval)
    if not (0 <= s0 < s1 <= 1 and a0 < a1):
        raise ValueError("invalid transverse cell")
    count, reason = _certify_cell(field, b, (s0, s1, a0, a1), periods, subdivisions)
    return AtlasCell(
        (s0, s1),
        (a0, a1),
        (s1 - s0) * (a1 - a0),
        0 if count is None else count,
        count,
        reason,
        () if count is None else tuple(range(count)),
    )


def build_atlas(
    field: BoozerFieldLike, b: float, config: AtlasConfig = AtlasConfig()
) -> BranchAtlas:
    """Build owned samples and conservative finite-window cells (§8).

    Each chart owns entry roots with 0 <= physical scan distance < one field
    period. This half-open convention removes duplicate lifted entries. Unknown
    cells have no assigned exact multiplicity or K-weight.
    """
    if not np.isfinite(b) or b <= 0:
        raise ValueError("b must be finite and positive")
    s_grid = np.linspace(0, 1, config.n_s)
    alpha_grid = np.linspace(0, 2 * np.pi, config.n_alpha, endpoint=False)
    samples = []
    for s in s_grid:
        for alpha in alpha_grid:
            try:
                wells, reason = _owned(
                    _line(field, float(s), float(alpha), config.periods),
                    b,
                    config.compute_actions,
                )
            except (ValueError, ArithmeticError) as error:
                wells, reason = (), str(error)
            samples.append(AtlasSample(float(s), float(alpha), wells, reason))
    cells = []
    da = 2 * np.pi / config.n_alpha
    for i in range(config.n_s - 1):
        for j in range(config.n_alpha):
            s0, s1 = map(float, s_grid[i : i + 2])
            a0, a1 = float(alpha_grid[j]), float(alpha_grid[j] + da)
            cell = classify_cell(
                field, b, (s0, s1), (a0, a1), config.periods, config.cell_subdivisions
            )
            if cell.multiplicity_upper is not None:
                vertices = (
                    samples[i * config.n_alpha + j],
                    samples[i * config.n_alpha + (j + 1) % config.n_alpha],
                    samples[(i + 1) * config.n_alpha + j],
                    samples[(i + 1) * config.n_alpha + (j + 1) % config.n_alpha],
                )
                if any(len(v.wells) != cell.multiplicity_upper for v in vertices):
                    cell = AtlasCell(
                        cell.s_interval,
                        cell.alpha_interval,
                        cell.area,
                        0,
                        None,
                        "sampled root count disagrees with cell certificate",
                    )
            cells.append(cell)
    return BranchAtlas(field, float(b), tuple(samples), tuple(cells), config)


def refine_atlas_cell(
    atlas: BranchAtlas,
    cell_index: int,
    s_interval: tuple[float, float],
    alpha_interval: tuple[float, float],
    subdivisions: int = 12,
) -> BranchAtlas:
    """Replace one cell by disjoint local charts around a target (§8.2–8.4).

    The target and its complement partition the former cell. Reclassifying
    all pieces preserves unique integration ownership; a tiny certified patch
    does not erase the unknown weight or links in its complement.
    """
    if not 0 <= cell_index < len(atlas.cells):
        raise IndexError("cell_index out of range")
    parent = atlas.cells[cell_index]
    s0, s1 = map(float, s_interval)
    a0, a1 = map(float, alpha_interval)
    if not (
        parent.s_interval[0] <= s0 < s1 <= parent.s_interval[1]
        and parent.alpha_interval[0] <= a0 < a1 <= parent.alpha_interval[1]
    ):
        raise ValueError("target must lie inside the selected chart cell")
    s_cuts = sorted(set((*parent.s_interval, s0, s1)))
    a_cuts = sorted(set((*parent.alpha_interval, a0, a1)))
    pieces = []
    samples = list(atlas.samples)
    sample_keys = {(sample.s, sample.alpha % (2 * np.pi)) for sample in samples}
    for left_s, right_s in zip(s_cuts[:-1], s_cuts[1:]):
        for left_a, right_a in zip(a_cuts[:-1], a_cuts[1:]):
            cell = classify_cell(
                atlas.field,
                atlas.b,
                (left_s, right_s),
                (left_a, right_a),
                atlas.config.periods,
                subdivisions,
            )
            if cell.multiplicity_upper is not None:
                corners = []
                for s in (left_s, right_s):
                    for alpha in (left_a, right_a):
                        alpha = alpha % (2 * np.pi)
                        try:
                            wells, reason = _owned(
                                _line(atlas.field, s, alpha, atlas.config.periods),
                                atlas.b,
                                atlas.config.compute_actions,
                            )
                        except (ValueError, ArithmeticError) as error:
                            wells, reason = (), str(error)
                        corners.append(len(wells))
                        if (s, alpha) not in sample_keys:
                            samples.append(AtlasSample(s, alpha, wells, reason))
                            sample_keys.add((s, alpha))
                if any(count != cell.multiplicity_upper for count in corners):
                    cell = AtlasCell(
                        cell.s_interval,
                        cell.alpha_interval,
                        cell.area,
                        0,
                        None,
                        "sampled root count disagrees with refined certificate",
                    )
            pieces.append(cell)
    cells = atlas.cells[:cell_index] + tuple(pieces) + atlas.cells[cell_index + 1 :]
    if not np.isclose(sum(c.area for c in pieces), parent.area, rtol=0, atol=1e-12):
        raise ArithmeticError("refined chart partition lost transverse area")
    return BranchAtlas(atlas.field, atlas.b, tuple(samples), cells, atlas.config)


def height_at(
    field: BoozerFieldLike,
    s: float,
    alpha: float,
    zeta_guess: float,
    bracket: float = 0.1,
) -> HeightSample:
    """Continue a nondegenerate extremum and evaluate H_j, ∇H_j (§10.1).

    The derivative is exact for the represented field; no field error bound is
    implied. ``zeta_guess`` must select the intended local root family.
    """
    iota = float(field.iota(s))

    def D(z):
        return float(field.D_B(s, alpha + iota * z, z))

    zeta = float(zeta_guess)
    if abs(D(zeta)) > 1e-11:
        left, right = zeta - bracket, zeta + bracket
        if D(left) * D(right) >= 0:
            raise ValueError("extremum continuation bracket failed")
        zeta = brentq(D, left, right, xtol=1e-12)
    theta = alpha + iota * zeta
    curvature = float(field.D2_B(s, theta, zeta))
    if not np.isfinite(curvature) or abs(curvature) < 1e-10:
        raise ValueError("degenerate height has no smooth local gradient")
    if not hasattr(field, "diota_ds"):
        raise ValueError("analytic radial shear unavailable for local height gradient")
    dtheta = float(field.dB_dtheta(s, theta, zeta))
    return HeightSample(
        float(s),
        float(alpha),
        zeta,
        float(field.B(s, theta, zeta)),
        curvature,
        float(field.dB_ds(s, theta, zeta) + field.diota_ds(s) * zeta * dtheta),
        dtheta,
    )


def _action(field, b, s, alpha, z0, z1, breakpoints):
    iota = float(field.iota(s))
    C = abs(float(field.C(s)))

    def integrand(z):
        B = float(field.B(s, alpha + iota * z, z))
        if B > b + 1e-10 * max(abs(b), 1.0):
            raise ValueError("B>b inside a purported limiting trapped well")
        return C / B * np.sqrt(max(0.0, 1 - B / b))

    interior = [z for z in breakpoints if min(z0, z1) < z < max(z0, z1)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        value, error = quad(
            integrand,
            min(z0, z1),
            max(z0, z1),
            points=interior,
            epsabs=1e-9,
            epsrel=1e-8,
            limit=150,
        )
    return float(value), float(error)


def transition_at(
    field: BoozerFieldLike,
    b: float,
    s: float,
    alpha: float,
    periods: int = 2,
    tolerance_B: float = 1e-8,
) -> AtlasTransition:
    """Locate limiting ports at a given physical marginal event parameter (§10.2).

    Simultaneous maxima return all contiguous limiting wells as possible ports,
    marked unknown. Actions are quadrature estimates; K is divergent at an exact
    generic maximum and is intentionally not fabricated.
    """
    L = 2 * np.pi / field.nfp
    sigma = float(np.sign(field.C(s)))
    z0 = -sigma * L
    scan = ForwardLineCatalogue(
        field,
        s,
        alpha + float(field.iota(s)) * z0,
        z0,
        ForwardScanConfig(max_periods=periods),
    )
    scan.extend_to(periods)
    maxima = sorted(
        (
            item
            for item in scan.extrema
            if item.kind == -1 and abs(item.B - b) <= tolerance_B and 0 <= item.zeta < L
        ),
        key=lambda item: item.u,
    )
    if not maxima:
        return AtlasTransition(
            "unknown", (s, alpha), (), (), "no certified marginal maximum"
        )
    if any(item.kind == 0 and abs(item.B - b) <= tolerance_B for item in scan.extrema):
        return AtlasTransition(
            "unknown", (s, alpha), (), (), "degenerate marginal endpoint"
        )
    # Ordinary sign-changing roots enclose the tangent group. Tangent extrema
    # themselves are excluded; they are not first-return crossings.
    crossings = []
    for i in range(len(scan.u) - 1):
        fl, fr = scan.B_samples[i] - b, scan.B_samples[i + 1] - b
        if fl * fr < 0:
            u = brentq(lambda v: float(scan._B(v)) - b, scan.u[i], scan.u[i + 1])
            crossings.append((u, np.sign(float(scan._D(u)))))
    first, last = maxima[0].u, maxima[-1].u
    entries = [u for u, sign in crossings if sign < 0 and u < first]
    exits = [u for u, sign in crossings if sign > 0 and u > last]
    if not entries or not exits:
        return AtlasTransition(
            "unknown",
            (s, alpha),
            tuple(x.zeta for x in maxima),
            (),
            "outer ordinary roots not found within scan window",
        )
    a, d = max(entries), min(exits)
    if any(a < u < d for u, _ in crossings):
        return AtlasTransition(
            "unknown",
            (s, alpha),
            tuple(x.zeta for x in maxima),
            (),
            "additional ordinary barrier inside event family",
        )
    query = scan.query(b)

    def known_tangent_neighborhood(left, right):
        for item in maxima:
            curvature = abs(float(scan.field.D2_B(s, *scan.coordinates(item.u))))
            distance = max(abs(left - item.u), abs(right - item.u))
            if (
                scan._third_bound is not None
                and curvature > scan._third_bound * distance
            ):
                return True
        return False

    if any(
        a <= right and left <= d and not known_tangent_neighborhood(left, right)
        for left, right in query.unknown_cells
    ):
        return AtlasTransition(
            "unknown",
            (s, alpha),
            tuple(x.zeta for x in maxima),
            (),
            "possible hidden barrier away from marginal maxima",
        )
    vertices = [a] + [x.u for x in maxima] + [d]
    all_z = [float(scan.coordinates(u)[1]) for u in vertices]
    ports = []
    k = len(maxima)
    for left, right in combinations(range(k + 2), 2):
        if k == 1:
            role = {(0, 2): "parent", (0, 1): "child_1", (1, 2): "child_3"}[
                (left, right)
            ]
        else:
            role = f"possible_{left}_{right}"
        try:
            action, error = _action(
                field, b, s, alpha, all_z[left], all_z[right], all_z[1:-1]
            )
        except (ValueError, ArithmeticError, IntegrationWarning) as failure:
            return AtlasTransition(
                "unknown",
                (s, alpha),
                tuple(x.zeta for x in maxima),
                (),
                f"limiting action quadrature failed: {failure}",
            )
        ports.append(
            AtlasPort(
                role, (float(s), float(alpha)), all_z[left], all_z[right], action, error
            )
        )
    if k == 1:
        actions = {port.role: port for port in ports}
        parent, child_1, child_3 = (
            actions["parent"],
            actions["child_1"],
            actions["child_3"],
        )
        marginal_zeta = float(maxima[0].zeta)
        if any(
            abs(left - right) > 1e-10
            for left, right in (
                (child_1.zeta_out, marginal_zeta),
                (child_3.zeta_in, marginal_zeta),
                (child_1.zeta_in, parent.zeta_in),
                (child_3.zeta_out, parent.zeta_out),
            )
        ):
            return AtlasTransition(
                "unknown",
                (float(s), float(alpha)),
                tuple(x.zeta for x in maxima),
                (),
                "limiting port endpoints do not match the certified marginal maximum",
            )
        # The gap checks quadrature consistency for this partition. Independent
        # one-sided well actions are compared in the acceptance test (§10.2).
        gap = abs(parent.action_length - child_1.action_length - child_3.action_length)
        allowance = max(1e-8, 10 * sum(port.error_estimate for port in ports))
        if gap > allowance:
            return AtlasTransition(
                "unknown",
                (float(s), float(alpha)),
                tuple(x.zeta for x in maxima),
                tuple(ports),
                f"limiting action partition consistency failed: gap={gap:.3g}",
            )
    return AtlasTransition(
        "generic" if k == 1 else "multiway_unknown",
        (float(s), float(alpha)),
        tuple(x.zeta for x in maxima),
        tuple(ports),
        None if k == 1 else "simultaneous marginal maxima",
    )


def plot_atlas_diagnostics(atlas: BranchAtlas, event: AtlasTransition | None = None):
    """Plot chart multiplicity, owned entry lifts and event actions (§17.2–17.3).

    The image is diagnostic only. Gray cells have unknown multiplicity; their
    geometric area is not a K-weight or a reachability-error bound.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    ax = axes[0]
    for cell in atlas.cells:
        value = cell.multiplicity_upper
        color = "0.7" if value is None else plt.cm.viridis(min(value / 6, 1))
        ax.add_patch(
            Rectangle(
                (cell.alpha_interval[0], cell.s_interval[0]),
                cell.alpha_interval[1] - cell.alpha_interval[0],
                cell.s_interval[1] - cell.s_interval[0],
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
            )
        )
    ax.set(
        xlim=(0, 2 * np.pi),
        ylim=(0, 1),
        xlabel="chart alpha [rad]",
        ylabel="s",
        title="Certified count; gray = unknown",
    )
    ax = axes[1]
    L = 2 * np.pi / atlas.field.nfp
    for sample in atlas.samples:
        for well in sample.wells:
            ax.scatter(
                sample.alpha,
                well.zeta_in / L,
                c=well.branch_id,
                cmap="tab10",
                vmin=0,
                vmax=9,
                s=12,
            )
    ax.set(
        xlim=(0, 2 * np.pi),
        xlabel="chart alpha [rad]",
        ylabel="entry zeta / field period [lift]",
        title="Separate incoming roots",
    )
    ax = axes[2]
    if event is not None and event.ports:
        labels = [port.role for port in event.ports]
        actions = [port.action_length for port in event.ports]
        ax.bar(np.arange(len(labels)), actions)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
        ax.set(ylabel="A [length]", title=f"{event.status} event; same (s,alpha)")
    else:
        points = [
            (sample.alpha, well.action_length)
            for sample in atlas.samples
            for well in sample.wells
            if np.isfinite(well.action_length)
        ]
        if points:
            ax.scatter(*np.asarray(points).T, s=12)
        failed = [
            sample.alpha
            for sample in atlas.samples
            for well in sample.wells
            if well.action_status != "not_computed"
            and not np.isfinite(well.action_length)
        ]
        if failed:
            ax.scatter(
                failed,
                [0.04] * len(failed),
                transform=ax.get_xaxis_transform(),
                marker="x",
                color="tab:red",
                s=22,
                label="unresolved A",
            )
            ax.legend(loc="best")
        ax.set(
            xlabel="chart alpha [rad]",
            ylabel="A [length]",
            title="Sampled action (estimates)",
        )
    fig.suptitle(
        f"Root-labelled atlas | b={atlas.b:.8g} | "
        f"unknown area={atlas.unknown_area:.3g} | finite Fourier-model scope"
    )
    return fig
