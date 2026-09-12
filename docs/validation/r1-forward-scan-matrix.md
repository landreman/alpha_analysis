# R1 shared forward-scan evidence

This is a one-line-per-field R1 probe of all 30 physical file/pitch cases,
at two scan resolutions. It does not establish global atlas coverage,
population bounds, accessibility or final f acceptance.

Baseline: `ff3bfecd3d78302aecb3b5549fdb918b4a11ef6a`. Code SHA256: `e837ee181d8b3aa17c3bf717ea9042f512b4a1a6ccbe054424676d0f4da50750`.
Hardware: `macOS-14.7.6-arm64-arm-64bit`; one worker; new field objects, warm OS cache uncontrolled.
Source declaration: `h(rho)=1 (UniformSourceProfile); scans and A/K are source independent`.
Pitch source: R0 sampled radially global extrema estimates; not field bounds.
Numerical scope: Fourier-model root envelope; A/K n-versus-2n estimates; no field enclosure.
The scan uses s=0.5, theta0=zeta0=0. Each catalogue begins with four
field periods and resumes 4→8→16 while a queried pitch remains open at
the right boundary. The 16-period cap is a work limit, not passing proof.
One catalogue is shared over all six pitches per field and resolution.
The table reports complete wells only; incomplete roots are not assigned
zero action or classified passing.
coarse: 12 complete-window, 16 censored-window, 2 analytic passing-line probes, 56 finite A/K results
fine: 12 complete-window, 16 censored-window, 2 analytic passing-line probes, 56 finite A/K results
Unresolved B=b root cells across all probes: 0. Up to 28 cells per line still lack an extrema-completeness proof.
Measured shared scan+query: 3.838 s across both grids and all pitches; fresh catalogue scan+query per pitch: 11.189 s (2.91x cost ratio).
The shared total includes 2.366 s of incremental scan extensions; the JSON records each 4/8/16-period query snapshot and extension cost.
Initial scan time is repeated in each row for context; count it once
per field/resolution when summing shared wall time.
This comparison uses the same field objects and one line per field;
it excludes quadrature from both sides and is not a whole-equilibrium speed claim.
Maximum matched coarse/fine relative A or K difference: 1.542e-09 (resolution diagnostic only).
Complete-well integrals with a non-REGULAR status: 0. Maximum first-well relative difference from independent adaptive quadrature: 6.772e-10; from the legacy tracer on coarse probes: 2.444e-09.
The n3are field yielded 2 complete wells across both resolutions after resumption; higher-pitch windows that remain censored are retained as such.

| field | λn | level | periods | complete wells | status | unknown cells | initial+resume scan s | query+batch s |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 0.05 | coarse | 4→4 | 0 | REGULAR | 0 | 0.069 | 0.003 |
| 0 | 0.1 | coarse | 4→4 | 0 | REGULAR | 0 | 0.069 | 0.003 |
| 0 | 0.5 | coarse | 4→8 | 6 | MAX_PERIODS | 0 | 0.123 | 0.039 |
| 0 | 0.8 | coarse | 4→16 | 5 | MAX_PERIODS | 0 | 0.171 | 0.064 |
| 0 | 0.9 | coarse | 4→16 | 0 | MAX_PERIODS | 0 | 0.069 | 0.012 |
| 0 | 0.95 | coarse | 4→16 | 0 | MAX_PERIODS | 0 | 0.069 | 0.012 |
| 0 | 0.05 | fine | 4→4 | 0 | REGULAR | 0 | 0.108 | 0.005 |
| 0 | 0.1 | fine | 4→4 | 0 | REGULAR | 0 | 0.108 | 0.005 |
| 0 | 0.5 | fine | 4→8 | 6 | MAX_PERIODS | 0 | 0.215 | 0.047 |
| 0 | 0.8 | fine | 4→16 | 5 | MAX_PERIODS | 0 | 0.315 | 0.078 |
| 0 | 0.9 | fine | 4→16 | 0 | MAX_PERIODS | 0 | 0.108 | 0.023 |
| 0 | 0.95 | fine | 4→16 | 0 | MAX_PERIODS | 0 | 0.108 | 0.024 |
| 1 | 0.05 | coarse | 4→4 | 0 | REGULAR | 0 | 0.053 | 0.003 |
| 1 | 0.1 | coarse | 4→4 | 0 | REGULAR | 0 | 0.053 | 0.003 |
| 1 | 0.5 | coarse | 4→8 | 6 | MAX_PERIODS | 0 | 0.102 | 0.020 |
| 1 | 0.8 | coarse | 4→8 | 3 | MAX_PERIODS | 0 | 0.053 | 0.025 |
| 1 | 0.9 | coarse | 4→16 | 0 | MAX_PERIODS | 0 | 0.238 | 0.017 |
| 1 | 0.95 | coarse | 4→16 | 0 | MAX_PERIODS | 0 | 0.053 | 0.012 |
| 1 | 0.05 | fine | 4→4 | 0 | REGULAR | 0 | 0.107 | 0.005 |
| 1 | 0.1 | fine | 4→4 | 0 | REGULAR | 0 | 0.107 | 0.005 |
| 1 | 0.5 | fine | 4→8 | 6 | MAX_PERIODS | 0 | 0.211 | 0.029 |
| 1 | 0.8 | fine | 4→8 | 3 | MAX_PERIODS | 0 | 0.107 | 0.034 |
| 1 | 0.9 | fine | 4→16 | 0 | MAX_PERIODS | 0 | 0.321 | 0.033 |
| 1 | 0.95 | fine | 4→16 | 0 | MAX_PERIODS | 0 | 0.107 | 0.024 |
| 2 | 0.05 | coarse | 4→4 | 4 | REGULAR | 0 | 0.054 | 0.013 |
| 2 | 0.1 | coarse | 4→4 | 4 | REGULAR | 0 | 0.054 | 0.010 |
| 2 | 0.5 | coarse | 4→4 | 4 | REGULAR | 0 | 0.054 | 0.010 |
| 2 | 0.8 | coarse | 4→4 | 4 | REGULAR | 0 | 0.054 | 0.017 |
| 2 | 0.9 | coarse | 4→16 | 8 | MAX_PERIODS | 0 | 0.239 | 0.059 |
| 2 | 0.95 | coarse | 4→16 | 0 | MAX_PERIODS | 0 | 0.054 | 0.012 |
| 2 | 0.05 | fine | 4→4 | 4 | REGULAR | 0 | 0.106 | 0.015 |
| 2 | 0.1 | fine | 4→4 | 4 | REGULAR | 0 | 0.106 | 0.013 |
| 2 | 0.5 | fine | 4→4 | 4 | REGULAR | 0 | 0.106 | 0.014 |
| 2 | 0.8 | fine | 4→4 | 4 | REGULAR | 0 | 0.106 | 0.022 |
| 2 | 0.9 | fine | 4→16 | 8 | MAX_PERIODS | 0 | 0.433 | 0.076 |
| 2 | 0.95 | fine | 4→16 | 0 | MAX_PERIODS | 0 | 0.106 | 0.028 |
| 3 | 0.05 | coarse | 4→4 | 3 | REGULAR | 0 | 0.051 | 0.012 |
| 3 | 0.1 | coarse | 4→4 | 3 | REGULAR | 0 | 0.051 | 0.015 |
| 3 | 0.5 | coarse | 4→4 | 3 | MAX_PERIODS | 0 | 0.051 | 0.009 |
| 3 | 0.8 | coarse | 4→16 | 2 | MAX_PERIODS | 0 | 0.201 | 0.032 |
| 3 | 0.9 | coarse | 4→16 | 0 | MAX_PERIODS | 0 | 0.051 | 0.012 |
| 3 | 0.95 | coarse | 4→4 | 0 | NO_WELL | 0 | 0.051 | 0.000 |
| 3 | 0.05 | fine | 4→4 | 3 | REGULAR | 0 | 0.102 | 0.016 |
| 3 | 0.1 | fine | 4→4 | 3 | REGULAR | 0 | 0.102 | 0.013 |
| 3 | 0.5 | fine | 4→4 | 3 | MAX_PERIODS | 0 | 0.102 | 0.011 |
| 3 | 0.8 | fine | 4→16 | 2 | MAX_PERIODS | 0 | 0.441 | 0.047 |
| 3 | 0.9 | fine | 4→16 | 0 | MAX_PERIODS | 0 | 0.102 | 0.024 |
| 3 | 0.95 | fine | 4→4 | 0 | NO_WELL | 0 | 0.102 | 0.000 |
| 4 | 0.05 | coarse | 4→4 | 0 | REGULAR | 0 | 0.039 | 0.002 |
| 4 | 0.1 | coarse | 4→4 | 0 | REGULAR | 0 | 0.039 | 0.002 |
| 4 | 0.5 | coarse | 4→8 | 1 | MAX_PERIODS | 0 | 0.087 | 0.018 |
| 4 | 0.8 | coarse | 4→16 | 0 | MAX_PERIODS | 0 | 0.113 | 0.013 |
| 4 | 0.9 | coarse | 4→16 | 0 | MAX_PERIODS | 0 | 0.039 | 0.010 |
| 4 | 0.95 | coarse | 4→4 | 0 | NO_WELL | 0 | 0.039 | 0.000 |
| 4 | 0.05 | fine | 4→4 | 0 | REGULAR | 0 | 0.075 | 0.004 |
| 4 | 0.1 | fine | 4→4 | 0 | REGULAR | 0 | 0.075 | 0.004 |
| 4 | 0.5 | fine | 4→8 | 1 | MAX_PERIODS | 0 | 0.149 | 0.028 |
| 4 | 0.8 | fine | 4→16 | 0 | MAX_PERIODS | 0 | 0.219 | 0.027 |
| 4 | 0.9 | fine | 4→16 | 0 | MAX_PERIODS | 0 | 0.075 | 0.020 |
| 4 | 0.95 | fine | 4→4 | 0 | NO_WELL | 0 | 0.075 | 0.000 |

## Files and diagnostic plots

- 0: `boozmn_20260402-01-038_Ax_PCA_20dofs_allNfp_aspect6_eval000290_low_resolution.nc` SHA256 `49eec0d9ff88fb52cc377be7d190da35b6815210baa36c5daee2bea7f6ab3fbd`; load 0.005 s
  ![field 0 lifted scan](r1-forward-scan-plots/field-0.png)
- 1: `boozmn_20260402-01-178_TURBO_Garabedian_mpol1_xmin0p1_allNfp_aspect6_eval000155.nc` SHA256 `ec583795826f7b9cb213d0c63977ecc03b1e72e98bcc53de9a0621f17934202e`; load 0.004 s
  ![field 1 lifted scan](r1-forward-scan-plots/field-1.png)
- 2: `boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_aspect10_DMercFail_m0p3_eval000323_low_resolution.nc` SHA256 `43346b456ee4b2654b2d6d13e9d85837ab7051de1b015b5a763c55f8007bbda7`; load 0.004 s
  ![field 2 lifted scan](r1-forward-scan-plots/field-2.png)
- 3: `boozmn_d23p4_tm_ns51_mbooz16_nbooz16.nc` SHA256 `c99156e12f828b58798b959d7768307aeb5415900f0860b4e3825156bfc4f36c`; load 0.005 s
  ![field 3 lifted scan](r1-forward-scan-plots/field-3.png)
- 4: `boozmn_n3are_R7.75B5.7_mbooz18_nbooz12.nc` SHA256 `1b2f8b945f3bee66c473b1c0335fd072e30e183303ebafc2d3776f1d1c29e660`; load 0.004 s
  ![field 4 lifted scan](r1-forward-scan-plots/field-4.png)

The JSON companion retains all controls, root/window flags, unverified
extrema cells, each completed well and its A/K estimate, adaptive reference
and legacy first-well timing. The adaptive and GL error fields are numerical
estimates, not rigorous quadrature or field enclosures.
