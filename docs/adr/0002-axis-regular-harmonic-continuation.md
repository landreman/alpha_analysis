# ADR 0002: Axis-regular coefficients by harmonic continuation below the innermost surface

- **Status:** Decided.
- **Date:** 2026-08-26
- **Milestone:** 5 (robustness follow-up on branch `more_robust_surfaces`)
- **Design sections:** §7.3 (axis behavior), §7.1 (field interface), §8.3

## Context

`BoozerField` interpolates the Fourier coefficients \(b_{mn}(s)\) with a cubic spline
on the half-grid \(s\in[\Delta s/2, 1-\Delta s/2]\) and extrapolates below the
innermost surface \(s_0=\Delta s/2\) without constraint. Physically each \(m\neq0\)
harmonic must vanish at the axis like \(\rho^{|m|}=s^{|m|/2}\) (§7.3); the
unconstrained extrapolation instead leaves them finite at \(s=0\), making \(|B|\)
multivalued at the axis — for the W7-X reference file the \(\theta\)-spread at the
axis is about \(2\times10^{-3}\) T.

This broke surface extraction: a background edge ending on the axis node can have its
endpoint sample on one side of \(B=b\) while the limit along the edge is on the other,
so the marching bracket has no root and edge polishing correctly raised
`ROOT_FAILURE` ("edge-root residual exceeds B_tolerance"). Because the axis value of
\(B\) varies with \(\zeta\) over roughly \([2.33, 2.84]\), levels across most of the
pitch range could fail whenever an axis node fell in the discontinuity window.
§7.3 already mandates the fix ("implement and test an axis-regular radial
interpolation that enforces the expected harmonic scaling", option 1, preferred), but
does not settle the interpolation scheme.

## Options

1. **Continuation below \(s_0\) only (implemented)** — for \(s<s_0\), replace the
   spline value of each \(m\neq0\) coefficient by \(b_{mn}(s_0)(s/s_0)^{|m|/2}\)
   (its \(s\)-derivative analytically), and keep the spline for \(m=0\). \(B\)
   becomes single-valued at the axis and \(C^0\) at \(s_0\); values at \(s\ge s_0\)
   — everywhere the file has data — are bit-identical to before, so existing
   reference tests (`test_B_reference`, `test_get_min_max` at \(10^{-13}\)) are
   untouched. Cost: \(\partial_s\) of the coefficients is discontinuous at the
   ring \(s=s_0\), and the \(|m|=1\) derivative is genuinely singular at \(s=0\)
   (it evaluates to `inf` rather than a finite substitute).
2. **Re-spline globally in a transformed variable** (spline \(b_{mn}/s^{|m|/2}\) or
   spline in \(\rho\)) — smooth everywhere including at \(s_0\), but changes field
   values on every surface, breaking the \(10^{-13}\) reference values and the
   backward-compatibility requirement on existing public functions.
3. **Exclude \(s<s_{\min}\) with an error bound (§7.3 option 2)** — allowed as an
   intermediate, but every extraction result would carry an omitted-core bound, and
   the design names option 1 as the preferred final implementation.

## Decision

Option 1 decided.

## Consequences

- `BoozerField.B` is single-valued at the axis and continuous across \(s_0\);
  `bmnc`/`bmns`, `compute_B`, `get_min_max`, and all `_evaluate_fourier` users share
  the same continuation (`_evaluate_coefficient_spline` in
  `alpha_analysis/boozer_field.py`).
- `test_B_is_axis_regular` covers single-valuedness, continuity at \(s_0\), the
  \(\sqrt s\) scaling of the \(m\)-odd part, and derivative consistency.
- The milestone-5 note in `docs/STATUS.md` that required §7.3 work before axis
  topology enters a production result is discharged for the interpolation half; the
  finite-difference axis gradient in `surface_extract._logical_B_gradient` remains,
  but now differences a continuous field.
- Callers evaluating `dB_ds` exactly at \(s=0\) receive `inf` for \(|m|=1\) content
  (the physical singularity of the \(s\) parametrization) instead of a finite
  extrapolation artifact.
