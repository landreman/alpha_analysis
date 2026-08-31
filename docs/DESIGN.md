# Design: Computing the phase-space fraction connected to the plasma edge by constant-\(J\) contours and trapping-class transitions

**Repository:** `https://github.com/landreman/alpha_analysis`  
**Document role:** Technical design and implementation plan for AI-agent-driven development  
**Primary result:** The topological accessibility fraction \(f\) defined below  
**Primary implementation language:** Python 3.10+  

---

## Document map

- Sections 1–6 define the physical metric, well-state geometry, transitions, and full numerical pipeline.
- Sections 7–13 specify field evaluation, meshes, well tracing, reachability, and quadrature.
- Sections 14–19 specify package layout, public APIs, configuration, visualization, persistence, and dependencies.
- Sections 20–22 specify testing, error control, parallelism, performance, and the test-time budget.
- Sections 23–28 divide the project into agent-sized milestones and define scientific completion criteria.

The repository description in Section 2 reflects the tree inspected on 2026-08-24. Implementing agents must re-check the current tree and tests before modifying it.

## 1. Purpose and scope

This document specifies a practical algorithm and software architecture for computing the fraction of alpha-particle birth phase space that is connected to the plasma boundary by contours of the longitudinal adiabatic invariant

\[
J = \oint v_{\parallel}\,d\ell,
\]

when discontinuous changes in \(J\) are permitted at transitions between trapping classes.

The intended use is a general three-dimensional stellarator magnetic field represented in straight-field-line Boozer coordinates. The implementation must enumerate **all** trapped wells, including:

- wells that span several field periods;
- several independent wells on one field line in one field period;
- wells whose identities permute as the field-line label or radius changes;
- generic \(1\leftrightarrow 2\) split/merge transitions;
- later, nongeneric or symmetry-enforced multiway transitions.

The implementation must also provide extensive visual diagnostics. Every major geometric or topological object should be inspectable:

- the background volume mesh;
- the surfaces \(B=B_b\);
- their incoming and outgoing halves;
- entry and exit bounce points;
- \(J\), bounce time, extrema, winding, and itinerary labels on the surfaces;
- the marginal curves \(\Gamma_{\min}\) and \(\Gamma_{\max}\);
- transition curves and their branch correspondences;
- cut trapping sheets;
- constant-\(J\) contours;
- the edge-reachable region \(\Theta=1\);
- per-triangle quadrature contributions;
- the outer \(B_b\) integrand and convergence history.

This document is self-contained. It defines the physical metric, mathematical state space, numerical algorithm, data structures, APIs, tests, diagnostics, dependencies, persistence format, and a sequence of implementation milestones suitable for separate AI-agent pull requests.

### 1.1 What this metric does and does not represent

The indicator in this project is an **existential topological accessibility indicator**. It asks whether at least one path exists from a trapped state to \(\rho=1\), using:

1. connected constant-\(J\) contours on a continuously varying trapping sheet; and
2. all allowed branches at a trapping-class transition.

It is not, by itself, a physical loss probability. Near a separatrix, a real orbit can have phase-dependent or probabilistic capture into outgoing branches. Treating every allowed branch as accessible generally gives an upper envelope of physical collisionless loss. The data structures should leave room for a future directed or probabilistic transition model, but that extension is outside version 1.

### 1.2 Non-goals for version 1

Version 1 does not need to include:

- finite-orbit-width guiding-center trajectories;
- collisions, slowing down, or energy evolution;
- radial electric fields or electrostatic potential variation;
- probabilistic separatrix capture;
- transitions through passing trajectories followed by retrapping;
- automatic differentiation of \(f\);
- GPU support;
- multi-node MPI;
- an explicit Reeb graph or Reeb space;
- exact treatment of every nongeneric degenerate transition.

Nongeneric cases must be detected and reported rather than silently misclassified.

---

## 2. Existing repository and compatibility requirements

The repository currently contains:

- `alpha_analysis/boozer_field.py`: `BoozerField` and `BoozerSurface`, including loading `boozmn` or `wout` data, radial interpolation of \(G\), \(I\), \(\iota\), and Fourier coefficients, and evaluation of \(B\);
- `alpha_analysis/bounce_points.py`: a one-dimensional field-line scan that finds the contiguous allowed well nearest a selected toroidal center and optionally root-refines its bounce points;
- `alpha_analysis/J_invariant.py`: computation and plotting of one selected well’s normalized action, commonly using a center near \(\zeta=\pi/N_{\mathrm{fp}}\);
- tests for these modules and W7-X reference data.

These capabilities are useful and must remain operational. In particular, the current `compute_J_invariant()` result is a valuable regression comparison for a well selected by the existing heuristic. However, that heuristic does not enumerate every well and must not be used as the production well-bookkeeping method for the new metric.

The new implementation should be placed primarily in a subpackage:

```text
alpha_analysis/j_connectivity/
```

Existing public functions and command-line entry points should remain backward compatible unless a separate deliberate deprecation is approved.

Repository-specific implementation rules:

- use the existing `20220806-03` conda environment's Python 3.10.5 interpreter rather
  than creating a new conda environment; create a clean project `.venv` from it without
  `--system-site-packages`. Do not use `20250627-01-libE`: its Python 3.13 `readline`
  extension segfaults during pytest capture;
- keep code straightforward and avoid unnecessary framework abstractions;
- format new Python code consistently with `black`;
- every scientific feature must include tests;
- no failed or unresolved well trace may be silently converted to \(\Theta=0\), zero weight, or `NaN` that is later ignored.

---

## 3. Physical definition of the metric

### 3.1 Source and phase-space fraction

Let \(W=v^2/2\) be the specific kinetic energy and let \(W_0\) be the alpha birth specific energy. The isotropic monoenergetic source is

\[
S(\mathbf x,\mathbf v)
= \delta(W-W_0)\,h(\rho),
\]

where \(\rho\in[0,1]\) labels flux surfaces and \(h(\rho)\ge 0\) is a source profile known only up to an overall normalization.

Define

\[
f=\frac{N}{D},
\]

with

\[
N=\int d^3x\int d^3v\,S\,\Theta,
\qquad
D=\int d^3x\int d^3v\,S.
\]

The denominator is

\[
D
=4\pi\sqrt{2W_0}
\int_0^1 d\rho\,h(\rho)\frac{dV}{d\rho}.
\]

Let

\[
v_0=\sqrt{2W_0}.
\]

### 3.2 Coordinates

Use straight-field-line Boozer coordinates \((s,\theta,\zeta)\), where

\[
s=\rho^2=\frac{\psi}{\psi_{\mathrm{edge}}},
\qquad
\alpha=\theta-\iota(s)\zeta,
\]

and \(2\pi\psi\) is the toroidal flux. The magnetic field can be written locally as

\[
\mathbf B=\nabla\psi\times\nabla\alpha.
\]

The toroidal coordinate is periodic over one field period,

\[
0\le \zeta < L_\zeta,
\qquad
L_\zeta=\frac{2\pi}{N_{\mathrm{fp}}},
\]

and \(0\le\theta<2\pi\).

The implementation should work on the one-field-period quotient. Numerator and denominator then both acquire the same factor \(1/N_{\mathrm{fp}}\), which cancels in \(f\).

### 3.3 Magnetic moment and bounce field

The magnetic moment is

\[
\mu=\frac{v_\perp^2}{2B}.
\]

At a bounce point, \(v_\parallel=0\), so

\[
W_0=\mu B_b,
\qquad
B_b=\frac{W_0}{\mu}.
\]

The implementation must use \(B_b\), or equivalently \(\mu\), as the conserved pitch variable. It must not follow contours at fixed radius-dependent normalized pitch such as

\[
\lambda_n=\frac{B_b-B_{\min}(\rho)}{B_{\max}(\rho)-B_{\min}(\rho)},
\]

because constant \(\lambda_n\) is generally not constant \(\mu\) when the local extrema vary radially.

Passing particles have \(\Theta=0\). For \(B_b\) greater than the global maximum of \(B\), no bounce occurs and the state is passing.

### 3.4 Definition of \(\Theta\)

At fixed \(B_b\), each ordinary trapped well is a maximal field-line interval on which

\[
B<B_b.
\]

A trapped state is represented by its **incoming bounce point**, defined using the orientation along \(+\mathbf B\):

\[
B(\mathbf x_-)=B_b,
\qquad
\mathbf b\cdot\nabla B(\mathbf x_-)<0.
\]

Starting from \(\mathbf x_-\), follow \(+\mathbf B\) until the next outgoing crossing \(\mathbf x_+\):

\[
B(\mathbf x_+)=B_b,
\qquad
\mathbf b\cdot\nabla B(\mathbf x_+)>0.
\]

The state is edge connected if there exists a finite sequence of the following operations that reaches \(\rho=1\):

1. move within a connected component of a constant-\(J\) contour on one continuous trapping sheet;
2. at a trapping-class transition, jump to every well branch permitted by the split/merge relation, adopting the corresponding new value of \(J\), and continue on a constant-\(J\) contour on that branch.

Then

\[
\Theta=
\begin{cases}
1, & \text{if such a path reaches }\rho=1,\\
0, & \text{otherwise.}
\end{cases}
\]

The transition relation is undirected for this metric. Passing motion is not a bridge between trapped states: if a trapped branch terminates by becoming passing, that path terminates.

States exactly on critical curves have zero phase-space measure. For visualization, assign them the closure convention

\[
\Theta=1
\quad\text{if any incident regular branch is edge connected}.
\]

---

## 4. The space of wells at fixed \(B_b\)

### 4.1 Incoming-bounce surface

For fixed \(b=B_b\), define

\[
\Sigma_b^-=
\left\{
(s,\theta,\zeta):
B(s,\theta,\zeta)=b,
\quad
\mathbf b\cdot\nabla B<0
\right\}.
\]

Every regular point of \(\Sigma_b^-\) corresponds to exactly one trapped well. Therefore:

- a long well extending across many field periods is one point on \(\Sigma_b^-\);
- several wells on one field line correspond to several distinct points on \(\Sigma_b^-\);
- no global integer “well number” is needed;
- the complicated periodicity of \(\alpha\) is avoided;
- changes from QI-like to QA- or QH-like winding appear as ordinary topology and homology of curves on \(\Sigma_b^-\).

The production algorithm is organized by independently processed \(B_b\) slices. A shared three-dimensional background mesh is reused, but each \(\Sigma_b^-\) receives its own extracted and adaptively refined triangular mesh.

### 4.2 Action and bounce-time quantities

For a well from entry \(\ell_-\) to exit \(\ell_+\), define the half-bounce action length

\[
A
=
\int_{\ell_-}^{\ell_+}
\sqrt{1-\frac{B}{b}}\,d\ell,
\]

and the half-bounce time length

\[
K
=
\int_{\ell_-}^{\ell_+}
\frac{d\ell}{\sqrt{1-B/b}}.
\]

Then the physical full-bounce longitudinal invariant and full bounce time are

\[
J=2v_0 A,
\qquad
\tau_b=\frac{2K}{v_0}.
\]

At fixed \(W_0\), contours of \(J\), \(A\), or any global constant multiple of them are identical. The new code should store `action_length = A` as its authoritative action. For compatibility with the current plotting code, it may also expose

\[
J_{\mathrm{normalized}}=\frac{A}{L_{\mathrm{ref}}},
\qquad
L_{\mathrm{ref}}=R_{00}\frac{2\pi}{N_{\mathrm{fp}}},
\]

which matches the normalization used by the existing `compute_J_invariant()` implementation.

In Boozer coordinates, let

\[
C(s)=G(s)+\iota(s)I(s).
\]

Along a field line,

\[
\frac{d\theta}{d\zeta}=\iota(s),
\qquad
\frac{d\ell}{|d\zeta|}=\frac{|C(s)|}{B}.
\]

Hence

\[
A
=
\int_{\zeta_-}^{\zeta_+}
\frac{|C|}{B}
\sqrt{1-\frac{B}{b}}\,|d\zeta|,
\]

\[
K
=
\int_{\zeta_-}^{\zeta_+}
\frac{|C|}{B\sqrt{1-B/b}}\,|d\zeta|.
\]

The \(A\) integrand vanishes at an ordinary bounce point. The \(K\) integrand has an integrable inverse-square-root endpoint singularity. At a marginal maximum, \(K\) diverges logarithmically; this requires special quadrature treatment near \(\Gamma_{\max}\).

### 4.3 Surface measure and computational expression for \(f\)

The natural measure on the incoming-bounce surface is

\[
|d\psi\wedge d\alpha|.
\]

Using \(s=\psi/\psi_{\mathrm{edge}}\), the common factor \(\psi_{\mathrm{edge}}\) cancels between numerator and denominator. Define

\[
\omega=ds\wedge d\alpha.
\]

The numerator becomes

\[
N
=
2\pi\int d\mu
\int_{\Sigma_\mu^-}
h(\rho)\,\tau_b\,\Theta\,
|d\psi\wedge d\alpha|.
\]

Since

\[
|d\mu|=\frac{W_0}{b^2}\,db,
\]

and \(\tau_b=2K/v_0\), the dimensionless fraction can be evaluated without explicitly retaining \(W_0\) or \(\psi_{\mathrm{edge}}\):

\[
\boxed{
 f
 =
 \frac{
 \displaystyle
 \int_{B_{\min}^{\mathrm{global}}}^{B_{\max}^{\mathrm{global}}}
 \frac{db}{b^2}
 \int_{\Sigma_b^-}
 h(\sqrt{s})\,K\,\Theta\,|\omega|
 }{
 \displaystyle
 2\int_0^1 ds\,h(\sqrt{s})
 \int_0^{2\pi}d\theta
 \int_0^{2\pi/N_{\mathrm{fp}}}d\zeta\,
 \frac{|G+\iota I|}{B^2}
 }.
}
\]

Define the source-weighted normalized volume factor

\[
V_h
=
\int_0^1 ds\,h(\sqrt{s})
\int_0^{2\pi}d\theta
\int_0^{2\pi/N_{\mathrm{fp}}}d\zeta\,
\frac{|G+\iota I|}{B^2}.
\]

For a pitch slice, define

\[
Q(b)
=
\int_{\Sigma_b^-}
h(\sqrt{s})\,K\,\Theta\,|\omega|.
\]

Then

\[
f=\frac{1}{2V_h}\int \frac{Q(b)}{b^2}\,db.
\]

This is the primary computational formula.

### 4.4 Axis-regular representation of the surface measure

Use logical Cartesian coordinates in the poloidal disk:

\[
x=\rho\cos\theta,
\qquad
y=\rho\sin\theta,
\qquad s=x^2+y^2.
\]

The volume domain is the periodic solid cylinder

\[
x^2+y^2\le 1,
\qquad
0\le\zeta<L_\zeta,
\]

with the two end disks identified.

Although \(d\theta\) is singular at the axis, the two-form \(\omega\) is regular. Since

\[
d\alpha=d\theta-\iota(s)d\zeta-\iota'(s)\zeta\,ds,
\]

we have

\[
\omega
=ds\wedge d\theta-\iota(s)ds\wedge d\zeta.
\]

In \((x,y,\zeta)\),

\[
\boxed{
\omega
=2\,dx\wedge dy
-2\iota(s)
\left(x\,dx\wedge d\zeta+y\,dy\wedge d\zeta\right).
}
\]

For tangent vectors \(u\) and \(v\), evaluate

\[
\begin{aligned}
\omega(u,v)
={}&2(u_xv_y-u_yv_x)\\
&-\iota(s)
\left[
(2xu_x+2yu_y)v_\zeta
-(2xv_x+2yv_y)u_\zeta
\right].
\end{aligned}
\]

This formula should be used for triangle quadrature. It avoids constructing a globally single-valued \(\alpha\) and remains regular at \(x=y=0\).

---

## 5. Critical curves and trapping-class transitions

### 5.1 Parallel derivatives

Define the derivative along a field line at fixed \(s\):

\[
D_\parallel^{(\zeta)}
=\iota(s)\partial_\theta+\partial_\zeta.
\]

Then

\[
\mathbf b\cdot\nabla B
=\frac{B}{G+\iota I}
D_\parallel^{(\zeta)}B.
\]

The code must use the physical sign, including the sign of \(G+\iota I\). It must not assume that increasing \(\zeta\) is always the \(+\mathbf B\) direction.

At a marginal point on \(B=b\),

\[
D_\parallel^{(\zeta)}B=0.
\]

The sign of the second derivative

\[
D_\parallel^{(\zeta)2}B
\]

distinguishes a local minimum from a local maximum.

### 5.2 Boundary curves of \(\Sigma_b^-\)

The boundary can contain:

- `EDGE`: \(s=1\);
- `GAMMA_MIN`:
  \[
  B=b,
  \quad \mathbf b\cdot\nabla B=0,
  \quad D_\parallel^{(\zeta)2}B>0;
  \]
- `GAMMA_MAX`:
  \[
  B=b,
  \quad \mathbf b\cdot\nabla B=0,
  \quad D_\parallel^{(\zeta)2}B<0;
  \]
- `AXIS`: an intersection with \(s=0\), if present;
- `DEGENERATE`: points for which the second derivative is too small to classify reliably.

At `GAMMA_MIN`, a well is born or dies with \(A\to0\). Positive-action contours do not continue through it, so no transition hyperedge is added.

At `GAMMA_MAX`, a well split/merge occurs.

### 5.3 Generic split/merge relation

Let \(m\in\Gamma_{\max}\). Trace backward along \(-\mathbf B\) to the preceding incoming crossing \(a\). Trace forward from \(m\), ignoring the tangent contact itself, to the next ordinary outgoing crossing \(d\).

At the transition, three limiting wells meet:

- merged parent \(W=[a,d]\);
- first child \(w_1=[a,m]\);
- second child \(w_3=[m,d]\).

Their action lengths satisfy

\[
A_W=A_{w_1}+A_{w_3}.
\]

The curve of points \(a\) is the companion transition curve \(T\). Across \(T\), the first-return exit jumps, so \(A\) is multivalued in the geometric projection. The triangular surface mesh must be cut along \(T\), with duplicated vertices carrying the parent and child values separately.

The transition correspondence is pointwise in a common curve parameter \(u\):

\[
T_W(u)
\longleftrightarrow
T_{w_1}(u)
\longleftrightarrow
\Gamma_{w_3}(u).
\]

For the present existential definition, all three are mutually connected.

### 5.4 Multiway and degenerate transitions

The data model must support an arbitrary number of ports, even if the first implementation only resolves generic three-port events. A transition is represented as a hyperedge node with \(m\ge 3\) incident port curves.

If the code detects any of the following, it must refine or report an unresolved nongeneric event:

- \(|D_\parallel^{(\zeta)2}B|\) below tolerance;
- two or more marginal maxima at the same event parameter;
- a jump in itinerary larger than the generic one-maximum change;
- several \(\Gamma_{\max}\) components mapping to the same part of \(T\);
- failure of \(A_W\approx A_{w_1}+A_{w_3}\) under refinement.

Do not arbitrarily decompose a nongeneric event into binary transitions without recording that modeling choice.

---

## 6. High-level numerical algorithm

```mermaid
flowchart TD
    A[Load Boozer equilibrium] --> B[Build shared periodic background tetrahedral mesh]
    B --> C[Compute denominator V_h]
    B --> D[Choose adaptive B_b midpoint nodes]
    D --> E[Extract B = B_b surface]
    E --> F[Split by g = b dot grad B and retain incoming half]
    F --> F2[Downsample the incoming surface]
    F2 --> G[Trace every regular well; compute A, K, exit, extrema, itinerary]
    G --> H[Extract Gamma_min and Gamma_max]
    H --> I[Construct companion transition curves T]
    I --> J[Cut mesh into continuous-return sheets]
    J --> K[Direct contour tracer for validation]
    J --> L[Bounded action-atom flood fill from rho = 1]
    L --> M[Clip reachable parts of triangles]
    M --> N[Weighted surface quadrature Q(B_b)]
    N --> O[Adaptive midpoint integration in B_b]
    C --> P[Form f = integral Q/b^2 divided by 2 V_h]
    O --> P
    P --> Q[Convergence report, HDF5 checkpoint, plots and VTK files]
```

For each \(b\), the algorithm is:

1. Extract the level surface \(B=b\) from the shared tetrahedral mesh.
2. Split it along \(g=\mathbf b\cdot\nabla B=0\), retain \(g<0\), and preserve the \(g=0\) curves.
3. Typically downsample the incoming surface while preserving its topology and tagged boundaries.
4. At every regular surface vertex, trace the unique well and compute \(A\), \(K\), exit data, extrema, and a lifted return itinerary.
5. Adaptively refine where field interpolation, \(A\), \(K\), return itinerary, or critical curves are under-resolved.
6. Construct \(\Gamma_{\max}\) and the companion transition curve \(T\), cut the mesh, and add pointwise transition hyperedges.
7. Treat \(A\) as a continuous piecewise-linear scalar on each cut sheet.
8. Seed all \(A\)-contours that intersect `EDGE` as reachable.
9. Propagate reachable action intervals through ordinary triangles and transition hyperedges to a fixed point.
10. Integrate \(hK|\omega|\) only over the reachable sub-polygons of each triangle.
11. Return \(Q(b)\), diagnostics, unresolved-weight bounds, and visualization objects.

---

## 7. Magnetic-field evaluation requirements

### 7.1 General Fourier representation

The code must support stellarator-asymmetric fields, not only cosine modes. Represent

\[
B(s,\theta,\zeta)
=
\sum_k
\left[
B_k^c(s)\cos\chi_k
+B_k^s(s)\sin\chi_k
\right],
\]

with

\[
\chi_k=m_k\theta-n_k\zeta.
\]

The existing code reads and evaluates cosine coefficients. Before claiming general-stellarator support, add loading and evaluation of sine coefficients when present in `boozmn` or `booz_xform` output.

The field API must provide vectorized evaluation of:

- \(B\);
- \(\partial_s B\), when available;
- \(\partial_\theta B\);
- \(\partial_\zeta B\);
- \(D_\parallel^{(\zeta)}B\);
- \(D_\parallel^{(\zeta)2}B\);
- \(G(s)\), \(I(s)\), and \(\iota(s)\).

Analytic Fourier derivatives are required. Finite differences may be used only as tests or temporary fallbacks.

For each mode, with \(k=m\iota-n\),

\[
D_\parallel^{(\zeta)}B
=
\sum
\left[-kB^c\sin\chi+kB^s\cos\chi\right],
\]

\[
D_\parallel^{(\zeta)2}B
=
-\sum k^2
\left[B^c\cos\chi+B^s\sin\chi\right].
\]

### 7.2 Field protocol

Define a small protocol so synthetic test fields and future equilibrium backends can be used:

```python
class BoozerFieldLike(Protocol):
    nfp: int

    def B(self, s, theta, zeta) -> NDArray[np.float64]: ...
    def dB_ds(self, s, theta, zeta) -> NDArray[np.float64]: ...
    def dB_dtheta(self, s, theta, zeta) -> NDArray[np.float64]: ...
    def dB_dzeta(self, s, theta, zeta) -> NDArray[np.float64]: ...
    def D_B(self, s, theta, zeta) -> NDArray[np.float64]: ...
    def D2_B(self, s, theta, zeta) -> NDArray[np.float64]: ...
    def iota(self, s): ...
    def G(self, s): ...
    def I(self, s): ...
```

The existing `BoozerField` should implement this interface directly or through an adapter.

### 7.3 Axis behavior

The logical mesh should be axis regular, but the Fourier interpolation must also have a controlled axis limit. For a smooth scalar, the poloidal harmonic \(m\) should scale as \(\rho^{|m|}\).

The implementation must do one of the following before including axis-intersecting topology in a production result:

1. implement and test an axis-regular radial interpolation that enforces the expected harmonic scaling; or
2. exclude \(s<s_{\min}\), compute an explicit upper bound from the omitted source-weighted volume, and demonstrate convergence as \(s_{\min}\to0\).

The preferred final implementation is option 1. Option 2 is acceptable as an intermediate milestone, but the result object must report the omitted-core bound.

---

## 8. Mesh strategy

### 8.1 Authoritative numerical representation

All authoritative geometry and topology must be stored as NumPy arrays and integer IDs. PyVista, VTK, Gmsh, Matplotlib, and NetworkX objects are views or helpers, not the source of truth.

```python
@dataclass
class BackgroundMesh:
    points: FloatArray          # shape (n_points, 3), columns x, y, zeta
    tetrahedra: IntArray        # shape (n_tets, 4)
    periodic_node_pairs: IntArray
    boundary_tags: IntArray
    B: FloatArray
    D_B: FloatArray
    D2_B: FloatArray
```

### 8.2 Background mesh

The shared background mesh covers

\[
\{(x,y,\zeta):x^2+y^2\le1,\ 0\le\zeta\le L_\zeta\}
\]

with periodic identification of the end disks.

Provide two backends:

1. `StructuredPrismMeshBackend` for deterministic unit tests and early development;
2. `GmshBackgroundMeshBackend` for production-quality tetrahedra and spatially varying resolution.

The structured backend may triangulate a disk, extrude the triangles through uniformly spaced \(\zeta\) planes, and split each triangular prism into consistently oriented tetrahedra. The periodic seam must have matching nodes and connectivity.

The Gmsh backend should:

- mesh a logical cylinder, not the physical torus;
- declare the end disks periodic;
- export the periodic node map explicitly;
- support size fields near the axis, small \(|\nabla B|\), and known critical regions;
- close Gmsh after extracting plain arrays.

### 8.3 Surface extraction

Define a `SurfaceExtractor` interface. Implement in stages:

- a PyVista/VTK contour backend for rapid development and visualization;
- a custom marching-tetrahedra backend with parent-edge provenance for production.

The production extractor should:

1. find every tetrahedron intersected by \(B=b\);
2. locate each edge root, initially by linear interpolation and then by a bracketed scalar root solve along the background edge;
3. triangulate the intersection polygon consistently;
4. preserve the parent tetrahedron and parent edge IDs;
5. identify and merge periodic seam copies;
6. carry domain-boundary provenance so `EDGE` and `AXIS` can be tagged robustly.

The output is the full \(B=b\) surface. A second marching-triangles operation on the scalar \(g=\mathbf b\cdot\nabla B\) splits each triangle along \(g=0\), retains \(g<0\), and records the boundary polylines.

### 8.4 Surface refinement

A pitch slice must be independently refinable without modifying unrelated \(b\) slices.

During a calculation of \(f\), the incoming pitch surface should typically be
downsampled after extraction and the \(g=0\) split, but before bounce integrals
are evaluated at its vertices. The background volume mesh may need to be fine
enough to recover the correct level-set topology without the later surface
calculation needing to inherit its full triangle count. Prefer
topology-preserving shortest-edge collapse or an equivalent method that also
reduces the population of very small triangles and makes triangle sizes more
uniform. Downsampling must preserve every connected component and all `EDGE`,
`AXIS`, periodic-seam, `G_ZERO`, and unresolved-boundary provenance; moved
vertices must be projected back to \(B=b\), and no accepted collapse may change
the physical sign of \(g\) or invert a triangle. Bound drift in the scalar
axis-regular \(|ds\wedge d\alpha|\) measure both globally and separately on
every connected component during coarsening; triangle count alone is not an
accuracy criterion. A downsampling result must report the achieved triangle
count and rejection counts for its safety checks so a binding constraint is
visible. These scalar measure budgets prevent cancellation between components,
but they do not prevent local or within-component cancellation and do not
directly bound weighted bounce integrals. The requested reduction remains
subordinate to these invariants, later adaptive refinement, and convergence of
the final \(f\).

Downsampling occurs before \(A\), \(K\), and return-map data are attached to
vertices. Later adaptive refinement may add vertices back where those
quantities or the critical curves require more resolution. Never downsample
across a transition cut.

Refinement indicators include:

- geometric error of \(B=b\);
- nonlinear variation of \(A\) or \(K\);
- changes in lifted exit point or field-period count;
- changes in extrema sequence;
- proximity to \(\Gamma_{\min}\), \(\Gamma_{\max}\), or \(T\);
- mixed or unresolved reachability;
- large variation in the final weight \(hK|\omega|\).

New surface vertices must be projected back to \(B=b\). Use a safeguarded Newton or secant correction, with periodic coordinates unwrapped locally. Never project across a transition cut.

---

## 9. Well tracing and bounce quadrature

### 9.1 Trace input and output

A regular incoming surface point \(q_-=(s,\theta_-,\zeta_-)\) satisfies

\[
B(q_-)=b,
\qquad g(q_-)<0.
\]

The trace follows \(+\mathbf B\). Let

\[
\sigma_\zeta=\operatorname{sign}(G+\iota I).
\]

Then unwrapped \(\zeta\) advances with sign \(\sigma_\zeta\), and

\[
\theta(\zeta)=\theta_-+\iota(s)(\zeta-\zeta_-).
\]

Return a `WellTrace` record:

```python
@dataclass
class WellTrace:
    status: TraceStatus
    b: float
    q_in: FloatArray                 # reduced logical coordinates
    q_out_reduced: FloatArray
    zeta_out_unwrapped: float
    field_period_count: int
    action_length: float             # A
    bounce_time_length: float        # K
    extrema_zeta_unwrapped: FloatArray
    extrema_B: FloatArray
    extrema_kind: IntArray           # -1 maximum, +1 minimum
    n_internal_maxima: int
    itinerary_hash: np.uint64
    B_residual_in: float
    B_residual_out: float
    quadrature_error_A: float
    quadrature_error_K: float
```

### 9.2 Root search

Use a two-stage procedure:

1. **Scan:** sample \(F(\zeta)=B(s,\theta(\zeta),\zeta)-b\) along unwrapped \(\zeta\), beginning just inside the well. Detect the first regular negative-to-positive crossing in the \(+\mathbf B\) direction.
2. **Refine:** use `scipy.optimize.brentq` or a safeguarded in-house solver on the bracket.

The scan must also locate extrema by roots of

\[
D_\parallel^{(\zeta)}B=0.
\]

A nearly tangent contact with \(F=0\) and \(D_\parallel^{(\zeta)}B\approx0\) is a transition candidate, not an ordinary outgoing crossing.

The scan step must adapt to the highest retained Fourier mode or to local variation of \(B\). A fixed number of samples per field period is acceptable only with a demonstrated convergence study.

### 9.3 Long wells

Near the trapped-passing boundary, the exit may be many field periods away. The trace must retain unwrapped \(\zeta\) and may continue across arbitrary periodic seams.

A configurable maximum period count is allowed, but reaching it yields `TraceStatus.MAX_PERIODS`, not \(\Theta=0\). The result must include an upper bound or unresolved-weight estimate, and convergence must be checked by increasing the cap.

### 9.4 Endpoint-regularized quadrature

For ordinary roots, map the interval to \(t\in[-\pi/2,\pi/2]\), for example

\[
\zeta(t)=\frac{\zeta_-+\zeta_+}{2}
+\frac{\zeta_+-\zeta_-}{2}\sin t.
\]

The Jacobian vanishes at the endpoints and removes the inverse-square-root singularity from \(K\). Use adaptive Gauss-Kronrod quadrature or a fixed high-order rule with an error estimate.

The implementation should provide one shared evaluator that returns both \(A\) and \(K\) from the same field samples when practical.

### 9.5 Itinerary signature

`n_internal_maxima` is a useful transition diagnostic but is not a sufficient unique sheet label. The itinerary should include:

- lifted exit \(\zeta_+\);
- field-period count;
- reduced exit coordinates;
- ordered extrema kinds;
- ordered extrema positions modulo and beyond the field period;
- extrema heights relative to \(b\).

A stable hash may be formed from a quantized version for quick comparisons, while the unquantized arrays remain authoritative.

Adjacent vertices are on the same continuous-return sheet only when the lifted exit and extrema sequence vary continuously.

---

## 10. Transition construction and mesh cutting

### 10.1 Extract \(\Gamma_{\max}\)

After splitting \(\Sigma_b\) by \(g=0\), classify every `GAMMA` segment using \(D_\parallel^{(\zeta)2}B\). Refine any segment whose classification changes or is near zero.

Parameterize each connected \(\Gamma_{\max}\) polyline by cumulative arc length in the logical surface mesh. Preserve periodic seam continuity.

### 10.2 Construct the companion curve \(T\)

For sampled points \(m(u)\in\Gamma_{\max}\):

1. trace backward along \(-\mathbf B\);
2. find the preceding regular incoming crossing \(a(u)\);
3. trace forward from \(m(u)\), past the tangent contact, to the next regular outgoing crossing \(d(u)\);
4. compute the three limiting actions
   \[
   A_1(u)=A[a,m],\quad
   A_3(u)=A[m,d],\quad
   A_W(u)=A[a,d];
   \]
5. verify
   \[
   A_W-A_1-A_3
   \]
   is within the transition tolerance.

The points \(a(u)\) form \(T\).

`TransitionMappingConfig.max_curve_samples` is a work budget, not a uniform
coarsening request. For a bounded run, start from a deterministic coarse subset
of the authoritative critical-curve vertices and recursively map the existing
vertex nearest the arc-length midpoint of each uncertified interval. Retain
every mapped vertex. Compare its companion/marginal geometry and all three
actions with interpolation from the interval endpoints; refine on geometric or
action disagreement, a detected interior-maximum itinerary/count change, `EDGE`
proximity, or near self-contact. Near `EDGE`, resolve the local interval down to
adjacent authoritative vertices. The near-self-contact trigger is span-relative:
split the interval, then reevaluate the threshold on its shorter children; it may
cease to trigger before adjacency. `None` maps every authoritative vertex.

The geometric test is
`error <= curve_geometry_atol + curve_geometry_rtol * interval_u_length`, and
each port's action test is
`error <= curve_action_atol + curve_action_rtol * max(abs(endpoint/midpoint A))`.
The absolute tolerances have logical-distance and action-length units
respectively. `curve_edge_proximity` is a normalized-flux distance `1-s` from
the plasma edge, and `curve_self_contact_ratio` scales a logical distance by the
current interval's arc length. These are reported controls, not universal
accuracy guarantees: certification is relative to the existing authoritative
critical-curve resolution and detected root-scan itinerary. Features below
either resolution remain §21.3 convergence work.

Every mapped midpoint is retained, replacing a rejected parent chord. Adjacent
authoritative vertices provide no further midpoint to test: reaching that terminal
PL resolution does not establish a continuous sub-vertex error bound.

If this certification cannot finish within the budget, return
`TransitionStatus.BUDGET_INSUFFICIENT`, retain finite mapped samples and explicit
uncertified source-index intervals, and do not cut. `sampling_samples_used`,
`authoritative_sample_count`, `sampling_certified`, `sampling_reason`, and the
largest midpoint discrepancies encountered make the budget outcome
machine-readable. Existing physical/numerical sample failures keep their own
statuses. `map_transitions_budget_sweep` reuses unique vertex traces with all
non-budget controls fixed, without changing any budget's retained sample set or
decision.

Early numerical/nongeneric stops retain the stopping interval and all other
uncertified source-index intervals. An explicit stop flag controls certification;
human-readable diagnostic wording does not control it. The physical/event reason
takes precedence over budget exhaustion when both occur on the last allowed sample.

`localize_transition_contacts` refines each count-change bracket on the true
`B=b, g=0` curve. Its default budget is 20 new midpoint traces per original
bracket, shared if an intermediate count splits that bracket; the final `u`
interval target is `1e-5`. Failed traces and exhausted intervals remain explicit.
Two endpoint rescans at twice the root-scan density, and a corrected midpoint,
may dissolve an alias only when the refined counts agree, the ordinary crossings
and actions remain within their existing solve/quadrature tolerances, and the
midpoint passes the geometry/action interpolation checks above. All probes are
retained. A below-`b` fold is not dismissed merely because the highest barrier
stays below `b`.

An equal-height event requires both maxima to satisfy `B=b`, `D_parallel B=0`,
and negative curvature on one lifted field line. Events shared by source curves
must match all marginal points and their lifted separations. Exactly sampled
nongeneric contacts also become explicit events; uncertain event geometry is
retained without inventing a location or a binary decomposition (§5.4).

`build_transition_arcs` subdivides at these events and computes one-sided limiting
actions by independent parent and child integrals. It preserves the source `u`
parameter and continuously aligns all field-line lifts. Each arc independently
certifies its remaining source intervals using the unused original source-vertex
budget. Localization traces have their separate budget. A failed endpoint build
retains the source mapping as diagnostic data and explicitly records the requested
unresolved `source_interval`; it does not claim finite limiting endpoint data or
permit a cut. `None` still means every authoritative source vertex is mapped,
not that continuous sub-vertex errors have been bounded.

### 10.3 Align \(T\) with the triangular mesh

Only a regular, sampling-certified transition may enter this operation. A
budget-insufficient transition remains an explicit unresolved hyperedge with
all ports and the budget reason; it is never ordinary missing connectivity.
This gate applies separately to each arc after milestone 10.2 localization.

The production implementation should insert \(T\) as a constrained polyline:

1. locate each sample in a surface triangle;
2. trace the polyline through crossed triangles;
3. split crossed edges and triangles;
4. snap or project new vertices to \(B=b\);
5. duplicate the final polyline vertices and edges;
6. assign parent action values to one copy and child-1 values to the other;
7. use the matched \(\Gamma_{\max}\) copy for child 3.

Insertion helpers may interpolate from a `T` vertex before that vertex has
received its limiting branch action. Preserve their interpolation dependency
chain and refresh off-cut descendants after port assignment, using each
stencil vertex's copy on the descendant's own sheet. Never retain a stale
parent/child blend. A stencil that genuinely crosses the final cut is
unresolved (`NaN` action), not an interpolation across the discontinuity;
later stages must account for that unresolved action under §21.2.

At event junctions, insert shared endpoint anchors before constraints. When a
folded field-line chart prevents the ordinary local insertion, a bounded path
through actual adjacent triangle faces may split crossed edges while retaining
component and cell provenance (ADR 0006). Existing constraints and physical
boundaries are barriers; each crossing must satisfy the existing local distance
allowance, and the path is limited to `max_corridor_faces=64`. This is local
constrained insertion, not global chart retriangulation. Degree checks prohibit
branches or dangling companion cuts away from `EDGE` or an explicit event.

The separate snap to an existing vertex on the projected segment uses a
chord-scaled physical offset allowance in normal-plane insertion:
`max_surface_distance_ratio * physical_chord_length`. It does not move that
vertex. This allowance can exceed the crossed-edge allowance when the anchor
budget leaves a long chord; preserve component provenance and require the same
connected-chain and decisive side-assignment checks. Report it as a remaining
surface-resolution control, not a local-edge error bound or convergence result.

Side assignment uses the existing decisive action comparison. Unrepresentable
arcs retain all ports and their reasons. If an uncut incident arc leaves different
one-sided event limits on one mesh vertex, retain those values on their distinct
ports and set that vertex's action to `NaN`, with its ID recorded in
`unresolved_event_action_vertex_ids`. Never overwrite one limit with another.
`CutSurface.unresolved_action_flux` measures every triangle containing unknown
action using §4.4's dimensionless `|ds wedge d alpha|`; this is not a bound on
`K`-weighted volume or on reachability. Event/transition connectivity uncertainty
remains separate. The NPZ format preserves arbitrary event-port endpoint
incidence, unknown-action vertex IDs, unresolved arc reasons, and insertion counts.
Matrix-level background/local refinement remains milestone 10.3 work.

For an earlier prototype, a mesh-aligned approximation based on itinerary changes across edges is acceptable, provided the direct backward map from \(\Gamma_{\max}\) is used to validate the location and branch correspondence.

### 10.4 Transition representation

```python
@dataclass
class TransitionPort:
    sheet_id: int
    polyline_vertex_ids: IntArray
    action_values: FloatArray
    role: Literal["parent", "child", "generic"]

@dataclass
class TransitionCurve:
    transition_id: int
    u: FloatArray
    ports: tuple[TransitionPort, ...]
    marginal_points: FloatArray
    additivity_residual: FloatArray
    status: TransitionStatus
```

Subdivide a transition curve at extrema of any port function \(A_p(u)\). On each resulting segment, every \(A_p(u)\) is piecewise linear and monotone or constant, which makes interval transfer unambiguous.

### 10.5 Sheet graph

After cutting, use union-find or mesh connectivity to assign `sheet_id` to each connected triangular component. Build a coarse NetworkX diagnostic graph:

- sheet nodes;
- transition nodes;
- incidence edges;
- attributes such as `touches_edge`, action range, area, and unresolved flags.

This coarse graph is useful for inspection but is **not sufficient to compute \(\Theta\)** because only some action contours on a sheet may reach the edge or a transition.

---

## 11. Reachability algorithms

Two reachability implementations are required:

1. a direct contour tracer used as a transparent correctness oracle;
2. a bounded interval/finite-atom flood fill used for production quadrature.

A continuous interval worklist is an optional accelerator, not the sole production correctness path.

### 11.1 Direct piecewise-linear contour tracer

On each cut sheet, \(A\) is continuous and piecewise linear. For a query point \(q\), let \(a=A(q)\). Within a triangle, the level set \(A=a\) is a line segment, a point, or a degenerate edge.

The tracer should:

1. locate the starting triangle;
2. determine the two edge intersections of \(A=a\);
3. walk across adjacent triangles;
4. use explicit periodic adjacency;
5. detect closure and already visited directed triangle-edge states;
6. report success on reaching `EDGE`;
7. at a transition port, map the event parameter \(u\) to every other port and recursively continue at each new action value;
8. terminate only when all branches have closed or ended without reaching the edge.

Memoize completed contour components when useful.

This method is too expensive for every quadrature point but is essential for tests and interactive debugging.

### 11.2 Interval set primitive

Implement a tested `IntervalSet` using sorted, disjoint closed intervals:

```python
class IntervalSet:
    def add(self, lo: float, hi: float) -> bool: ...
    def union(self, other: "IntervalSet") -> bool: ...
    def intersection(self, lo: float, hi: float) -> "IntervalSet": ...
    def affine_preimage(self, y0, y1, x0, x1) -> "IntervalSet": ...
    def affine_image(self, x0, x1, y0, y1) -> "IntervalSet": ...
```

Requirements:

- deterministic tolerance handling;
- no order dependence;
- correct constant-map behavior;
- compact serialization through `offsets` and `bounds` arrays;
- property-based tests for union, idempotence, monotonicity, and affine maps.

### 11.3 Interval flood fill on triangles

For each cut triangle \(c\), store

\[
L(c)\subseteq[A_{\min}(c),A_{\max}(c)],
\]

a union of action intervals whose local contours are edge connected.

#### Seed

For every triangle adjacent to an `EDGE` boundary edge \(e\), add the action range on that edge:

\[
I_e=[\min(A_i,A_j),\max(A_i,A_j)].
\]

#### Ordinary propagation

For an ordinary edge \(e\) shared by triangles \(c\) and \(c'\), propagate

\[
L(c')\leftarrow
L(c')\cup\left(L(c)\cap I_e\right),
\]

and conversely. For a linear scalar on a triangle, every regular contour is a single segment, so this propagation is exact for the piecewise-linear interpolant.

Use a work queue containing triangles whose interval set has changed.

#### Transition propagation

Represent each monotone transition segment by a shared parameter interval \([u_0,u_1]\) and one affine action function per port.

Maintain a reachable `IntervalSet` in \(u\) for the transition segment. When a port-adjacent triangle gains reachable action values:

1. intersect with the port edge’s action range;
2. take the full affine preimage in \(u\);
3. add it to the transition segment’s reachable \(u\)-set.

When the transition’s \(u\)-set changes:

1. map it forward through every incident port’s action function;
2. add the resulting action intervals to the adjacent triangles.

The existential rule is therefore logical OR over all ports.

#### Termination and production discretization

A continuous `IntervalSet` worklist is a useful fast algorithm, but it is not by itself a finite-termination guarantee. A cycle containing affine transition maps can generate an arbitrarily long sequence of new interval endpoints, even though the underlying triangular mesh is finite. Therefore version 1 must not base correctness only on the statement that all endpoints come from a finite set.

The production implementation must provide a **bounded finite-atom mode**:

1. Choose an action grid for each pitch slice or sheet and a parameter grid on every monotone transition segment.
2. Represent reachable sets by sparse bitmasks or runs of action/parameter atoms.
3. Whenever an exact interval is propagated, form two snapped images:
   - an **inner** image containing only atoms wholly contained in the exact interval;
   - an **outer** image containing every atom that intersects the exact interval.
4. Run the same worklist separately for the inner and outer masks. Both terminate because only finitely many bits can change.
5. Interpret the inner result as definitely reachable, the complement of the outer result as definitely unreachable, and the difference as unresolved.
6. Refine action and transition-parameter atoms wherever the unresolved phase-space weight is significant.

The continuous interval mode may be retained as an accelerator and exploratory diagnostic. It must have an iteration cap, report whether it stabilized, and be checked against the bounded finite-atom result and direct contour traces. The quoted value of \(f\) must come with lower and upper bounds obtained from the inner and outer reachable sets.

### 11.4 Bounded finite-atom flood fill

The finite-atom implementation should use the same physical propagation rules as the continuous interval method:

- ordinary triangle-to-triangle links preserve action;
- transition links preserve the common event parameter \(u\), not action;
- periodic identifications are explicit mesh adjacencies;
- no atom is connected merely because its numerical action range overlaps another disconnected component.

Use a global action grid per pitch slice for the first implementation because it simplifies bitmask operations and reproducibility. Later, allow per-sheet adaptive grids with explicit overlap maps. Initial grids may combine uniform bins with mandatory breakpoints at surface-vertex action values, transition extrema, and edge action extrema.

For every refinement level, save:

- lower and upper reachable masks;
- unresolved action intervals per triangle;
- unresolved transition-parameter intervals;
- the lower/upper contribution to \(Q(b)\);
- the change relative to the previous atom grid.

### 11.5 Critical PL cases

Adopt one consistent symbolic-perturbation convention for:

- contour values equal to a vertex value;
- equal action values on an edge;
- constant-action triangles;
- contours through a PL saddle vertex.

These values have measure zero, but inconsistent handling can create artificial finite-width connections. Unit tests must exercise them.

---

## 12. Surface quadrature

### 12.1 Reachable polygons

For triangle \(c\) and interval \([a_0,a_1]\subset L(c)\), clip the triangle by

\[
a_0\le A(q)\le a_1.
\]

Because \(A\) is linear in barycentric coordinates, clipping yields a convex polygon. A union of intervals yields a union of nonoverlapping polygons except at boundaries.

Do not approximate the contribution only as “reachable area fraction times one average weight” in the production calculation.

### 12.2 Weight

The slice weight is

\[
w(q)=h(\sqrt{s})K(q)|\omega|.
\]

For a triangle parameterized by edge vectors \(e_1,e_2\), evaluate \(|\omega(e_1,e_2)|\) using the axis-regular formula in Section 4.4.

### 12.3 Regular triangles

Away from marginal curves, interpolate \(K\) and other smooth factors from vertices or evaluate them at polygon quadrature points. Integrate over each clipped polygon using a standard triangle quadrature after fan triangulation.

### 12.4 Triangles adjacent to \(\Gamma_{\max}\)

At \(\Gamma_{\max}\), \(K\) diverges logarithmically. Do not store a finite fabricated vertex value.

For a boundary triangle:

1. set critical boundary-vertex `K` to `inf` or a dedicated status;
2. use interior quadrature nodes only;
3. evaluate or interpolate \(K\) using interior well traces;
4. recursively subdivide until the weighted integral converges;
5. optionally fit a local \(a+b\log d\) model in distance \(d\) from the marginal curve.

The product \(K|\omega|\) is integrable, but this must be demonstrated by refinement.

### 12.5 Per-slice result and uncertainty

Return:

- \(Q(b)\);
- a quadrature error estimate;
- lower and upper bounds from unresolved cells;
- contribution by sheet;
- contribution by radial bins;
- contribution by transition neighborhood versus regular region.

Unresolved cells are assigned \(\Theta=0\) for the lower bound and \(\Theta=1\) for the upper bound, with their weight integrated or conservatively bounded.

---

## 13. Denominator and outer \(B_b\) quadrature

### 13.1 Denominator

Compute

\[
V_h
=
\int_0^1 ds\,h(\sqrt{s})|C(s)|
\int_0^{2\pi}d\theta
\int_0^{2\pi/N_{\mathrm{fp}}}d\zeta\,B^{-2}.
\]

Recommended quadrature:

- Gauss-Legendre or adaptive quadrature in \(s\);
- periodic trapezoidal rules in \(\theta\) and \(\zeta\), exploiting spectral convergence for smooth Fourier data;
- independent resolution controls and convergence report.

This calculation should be implemented early and tested independently.

### 13.2 Global \(B\) bounds

The current coarse sampled minimum/maximum is not sufficient as the only production bound. Determine a safe bracket by:

1. taking extrema over a resolved background grid;
2. refining candidate extrema with local optimization in periodic coordinates;
3. adding a configurable safety margin tied to the interpolation error;
4. allowing explicit user-supplied bounds.

The outer rule evaluates only interior midpoint values, so degenerate endpoint surfaces need not be constructed.

### 13.3 Adaptive midpoint rule

The function

\[
F(b)=\frac{Q(b)}{b^2}
\]

may be continuous but nonsmooth at topology changes, and nongeneric cases may produce sharper behavior. Use a nested adaptive midpoint rule rather than relying on one high-order global Gaussian rule.

For interval \([b_0,b_1]\):

1. evaluate the midpoint \(m\);
2. compare the coarse midpoint estimate with the sum of two child midpoint estimates;
3. subdivide where the difference exceeds absolute and relative tolerances;
4. batch independent child evaluations for parallel execution;
5. reuse cached pitch slices.

Return the quadrature tree, estimated error, and the sampled \(F(b)\) curve.

### 13.4 Final result

```python
@dataclass
class FractionResult:
    f: float
    f_lower: float
    f_upper: float
    V_h: float
    numerator_integral: float
    b_nodes: FloatArray
    pitch_integrand: FloatArray
    pitch_error_estimates: FloatArray
    outer_error_estimate: float
    unresolved_weight: float
    metadata: RunMetadata
```

The code must verify

\[
0\le f_{\mathrm{lower}}\le f\le f_{\mathrm{upper}}\le1
\]

within numerical tolerance. Do not clamp a materially invalid value into this range; raise a diagnostic error instead.

---

## 14. Proposed package layout

```text
alpha_analysis/
    boozer_field.py                  # existing; extend Fourier and derivative support
    bounce_points.py                 # existing heuristic tools; preserve
    J_invariant.py                   # existing heuristic J plots; preserve
    j_connectivity/
        __init__.py
        config.py                    # dataclass configuration and validation
        types.py                     # array dataclasses, enums, protocols
        field.py                     # BoozerField adapter and derivative kernels
        synthetic_fields.py          # analytic fields for tests and demos
        background_mesh.py           # structured and Gmsh backends
        surface_extract.py           # B=b and g=0 extraction
        surface_refine.py            # projection and adaptive refinement
        well_trace.py                # roots, extrema, A, K, itinerary
        critical_curves.py           # EDGE, GAMMA_MIN/MAX, AXIS
        transitions.py               # T construction, cuts, hyperedges
        contour_trace.py             # direct PL validation tracer
        intervals.py                 # IntervalSet
        flood_fill.py                # production reachability
        polygon_clip.py              # A-interval clipping in triangles
        surface_quadrature.py         # Q(b)
        denominator.py               # V_h
        pitch_quadrature.py           # adaptive midpoint in b
        pipeline.py                   # public orchestration API
        io.py                         # HDF5 schema and restart
        visualization.py             # PyVista and Matplotlib diagnostics
        diagnostics.py               # consistency and convergence reports

test/
    test_field_derivatives.py
    test_background_mesh.py
    test_surface_extract.py
    test_well_trace.py
    test_critical_curves.py
    test_transitions.py
    test_intervals.py
    test_contour_trace.py
    test_flood_fill.py
    test_polygon_clip.py
    test_surface_quadrature.py
    test_denominator.py
    test_pitch_quadrature.py
    test_pipeline_synthetic.py
    test_pipeline_w7x.py

examples/
    plot_pitch_slice.py
    compute_j_connected_fraction.py
    inspect_transition.py
```

The exact number of files may be reduced if some remain short. Keep the conceptual boundaries even if small related modules are combined.

---

## 15. Public API

### 15.1 Main calculation

```python
def compute_j_connected_fraction(
    field: BoozerFieldLike,
    source_profile: Callable[[FloatArray], FloatArray] | None = None,
    config: JConnectivityConfig | None = None,
) -> FractionResult:
    """Compute the topological J-plus-transition edge-accessible fraction."""
```

`source_profile` takes \(\rho\), not \(s\). The default is a uniform profile.

### 15.2 Pitch slice

```python
def compute_pitch_slice(
    field: BoozerFieldLike,
    background: BackgroundMesh,
    b: float,
    source_profile: Callable[[FloatArray], FloatArray],
    config: PitchSliceConfig,
) -> PitchSliceResult:
    ...
```

### 15.3 Diagnostics and plotting

```python
def plot_background_mesh(...): ...
def plot_surface_quantity(slice_result, quantity: str, ...): ...
def plot_well_trace(trace: WellTrace, ...): ...
def plot_critical_curves(slice_result, ...): ...
def plot_transition(slice_result, transition_id: int, ...): ...
def plot_action_contours(slice_result, levels=None, ...): ...
def plot_reachability(slice_result, ...): ...
def plot_triangle_intervals(slice_result, triangle_ids, ...): ...
def plot_pitch_integrand(result: FractionResult, ...): ...
def write_vtk_bundle(slice_result, directory: Path): ...
```

### 15.4 Example use

```python
from alpha_analysis.boozer_field import BoozerField
from alpha_analysis.j_connectivity import (
    JConnectivityConfig,
    compute_j_connected_fraction,
)

field = BoozerField.from_boozmn("data/boozmn_example.nc")

config = JConnectivityConfig(
    output_directory="output/j_connectivity",
    save_pitch_slices=True,
    make_diagnostic_plots=True,
)

result = compute_j_connected_fraction(
    field,
    source_profile=lambda rho: 1.0 - rho**2,
    config=config,
)

print(result.f, result.f_lower, result.f_upper)
```

The Python API is primary. Add a simple command-line wrapper only after the API is stable.

---

## 16. Configuration

Use nested frozen dataclasses with explicit units and validation:

```python
@dataclass(frozen=True)
class WellTraceConfig:
    samples_per_field_period: int
    max_field_periods: int
    root_rtol: float
    root_atol_B: float
    incoming_root_max_offset: float  # radians
    extrema_tolerance: float
    quadrature_rtol: float
    quadrature_atol: float

@dataclass(frozen=True)
class SurfaceConfig:
    extractor: Literal["pyvista", "marching_tetrahedra"]
    max_refinement_levels: int
    B_surface_tolerance: float
    action_interpolation_tolerance: float
    itinerary_tolerance: float

@dataclass(frozen=True)
class ReachabilityConfig:
    mode: Literal["bounded_atoms", "continuous_intervals"]
    initial_action_atoms: int
    max_action_atoms: int
    initial_transition_atoms: int
    max_transition_atoms: int
    unresolved_weight_rtol: float
    unresolved_weight_atol: float
    interval_merge_rtol: float
    interval_merge_atol: float
    continuous_iteration_limit: int

@dataclass(frozen=True)
class PitchQuadratureConfig:
    b_min: float | None
    b_max: float | None
    initial_intervals: int
    max_intervals: int
    rtol: float
    atol: float
    n_jobs: int

@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool
    off_screen: bool
    save_png: bool
    save_pdf: bool
    save_vtk: bool
    selected_b_values: tuple[float, ...]

@dataclass(frozen=True)
class JConnectivityConfig:
    background: BackgroundMeshConfig
    surface: SurfaceConfig
    trace: WellTraceConfig
    reachability: ReachabilityConfig
    pitch: PitchQuadratureConfig
    visualization: VisualizationConfig
    output_directory: Path
    checkpoint_path: Path | None
```

Provide `development()`, `standard()`, and `high_accuracy()` constructors, but document that defaults are starting points, not certified accuracy settings.

---

## 17. Visualization and diagnostic requirements

Visualization is a first-class requirement, not an optional afterthought.

### 17.1 General rules

- Every mesh-like result must have a `to_pyvista()` conversion.
- Every scalar array must carry a human-readable name, units, and location (`point`, `cell`, `edge`, or `transition port`).
- Save VTK XML files (`.vtu` for volume meshes and `.vtp` for surfaces/curves) for ParaView.
- Provide static PNG output in headless CI mode.
- Keep plotting code separate from numerical kernels.
- Plot unresolved or failed data in a conspicuous separate category rather than hiding it.
- Include \(b\), equilibrium path/hash, mesh resolution, tolerances, and code commit in plot metadata or titles.

### 17.2 Required background-mesh views

1. Wireframe of the periodic solid cylinder.
2. Cutaway showing tetrahedra near the axis.
3. Periodic seam node pairs.
4. Point or cell colors for:
   - \(B\);
   - \(D_\parallel B\);
   - \(D_\parallel^2B\);
   - estimated interpolation error;
   - refinement level.
5. Histograms of tetrahedron quality and size.

### 17.3 Required pitch-surface views

For selected \(b\):

1. Full \(B=b\) surface.
2. Incoming \(g<0\) and outgoing \(g>0\) halves in different colors.
3. Disconnected component IDs.
4. Boundary curves:
   - `EDGE`;
   - `GAMMA_MIN`;
   - `GAMMA_MAX`;
   - `AXIS`;
   - `DEGENERATE`.
5. Surface colored by:
   - \(s\) or \(\rho\);
   - \(\theta\) and \(\zeta\);
   - \(A\) and legacy-normalized \(J\);
   - \(K\) or \(\log K\);
   - unwrapped exit \(\zeta_+\);
   - field-period count;
   - number of internal maxima;
   - itinerary ID;
   - sheet ID;
   - trace status;
   - action interpolation error;
   - additivity residual near transitions;
   - \(\Theta\);
   - per-cell contribution to \(Q(b)\).

### 17.4 Required well-trace diagnostics

For any selected surface vertex or clicked point:

1. \(B(\zeta)\) over the unwrapped trace, with \(b\) horizontal.
2. Entry, exit, maxima, minima, and tangent candidates marked.
3. The integrands for \(A\) and \(K\).
4. Cumulative \(A(\zeta)\) and \(K(\zeta)\).
5. The field-line path in \((\theta\bmod2\pi,\zeta\bmod L_\zeta)\).
6. The same path in unwrapped coordinates.
7. Text showing residuals, period count, and status.

### 17.5 Required transition diagnostics

For each transition:

1. \(\Gamma_{\max}(u)\) and \(T(u)\) on the surface.
2. Lines connecting matched parameter samples.
3. Parent, child-1, and child-3 action curves versus \(u\).
4. \(A_W-A_1-A_3\) versus \(u\).
5. The three corresponding \(B(\zeta)\) well profiles at selected \(u\).
6. A view of the mesh before and after cutting/vertex duplication.
7. Monotone subdivisions used by interval transfer.

### 17.6 Required reachability diagnostics

1. Directly traced constant-action contours for selected levels.
2. Inner, outer, and unresolved reachable action intervals per triangle.
3. Inner, outer, and unresolved transition \(u\)-intervals reached during flood fill.
4. Lower-bound, upper-bound, and resolved \(\Theta\) surfaces.
5. A comparison map of direct contour tracing versus flood fill at random query points.
6. Reachability work-queue iteration count and interval-count histogram.
7. Clipped reachable polygons in selected triangles.

### 17.7 Required global plots

1. \(Q(b)\) and \(Q(b)/b^2\) versus \(b\).
2. Adaptive midpoint nodes and interval error estimates.
3. Cumulative contribution to \(f\) versus \(b\).
4. Contribution versus \(\rho\).
5. Lower and upper bounds versus refinement level.
6. Convergence versus:
   - background resolution;
   - surface refinement;
   - field-line scan resolution;
   - maximum trace periods;
   - interval tolerance;
   - outer pitch tolerance.
7. Timing breakdown by pipeline stage.

---

## 18. Persistence and restart

Use HDF5 through `h5py`. One parent process should own the final file. Parallel workers should return compact results or write separate temporary files that the parent merges.

Suggested schema:

```text
/metadata
    schema_version
    equilibrium_path
    equilibrium_hash
    code_commit
    package_versions
    coordinate_convention
    config_json

/background_mesh
    points
    tetrahedra
    periodic_node_pairs
    boundary_tags
    B
    D_B
    D2_B

/denominator
    V_h
    quadrature_nodes_s
    convergence_history

/pitch_slices/<slice_key>
    b
    status
    /surface
        points
        triangles
        parent_tetrahedra
        boundary_edges
        boundary_tags
        sheet_id
    /vertex_data
        s
        theta
        zeta
        A
        K
        q_out_reduced
        zeta_out_unwrapped
        period_count
        n_internal_maxima
        itinerary_hash
        trace_status
    /extrema
        offsets
        zeta_unwrapped
        B
        kind
    /transitions
        transition_offsets
        u
        port_offsets
        port_vertex_ids
        port_action_values
        role
        additivity_residual
    /reachability
        action_atom_edges
        transition_atom_offsets
        transition_atom_edges
        triangle_inner_mask
        triangle_outer_mask
        transition_inner_mask
        transition_outer_mask
        unresolved_weight
        continuous_triangle_interval_offsets       # optional
        continuous_triangle_interval_bounds        # optional
        continuous_transition_u_interval_offsets   # optional
        continuous_transition_u_interval_bounds    # optional
        continuous_stabilized                       # optional
    /quadrature
        Q
        error_estimate
        lower_bound
        upper_bound
        triangle_contribution

/outer_quadrature
    b_nodes
    Q
    Q_over_b2
    interval_tree
    error_estimates

/result
    f
    f_lower
    f_upper
    numerator_integral
    V_h
```

Do not pickle PyVista or NetworkX objects. Reconstruct them from arrays.

Cache keys must include:

- equilibrium content hash;
- \(b\);
- all tolerances affecting the slice;
- mesh hash;
- code schema version.

---

## 19. Dependencies

### 19.1 Recommended first serious implementation

```text
numpy          authoritative arrays and vectorized operations
scipy          splines, root finding, quadrature, optimization
matplotlib     1D and 2D plots
pyvista / VTK  isosurfaces, inspection, VTK output, interactive 3D views
gmsh           production background tetrahedral mesh
networkx       coarse sheet/transition graph and diagnostics
numba          later acceleration of field-line scans and quadrature kernels
h5py           checkpointing and reproducible result files
joblib         parallel processing of B_b slices on one node
pytest         unit and integration tests
hypothesis     property-based tests for topology and interval operations
meshio         optional mesh-format interoperability
```

### 19.2 Dependency boundaries

```text
Gmsh / PyVista / NetworkX
        |
        v
plain NumPy arrays and integer IDs
        |
        v
NumPy / SciPy / Numba numerical kernels
```

PyVista objects must not enter Numba kernels. NetworkX must not duplicate every triangle as a Python graph node in the production flood fill.

### 19.3 Deferred dependencies

- Use Topology ToolKit only after the custom cut-sheet and contour algorithms are correct; it can then validate Reeb graphs on selected cases.
- Use `rustworkx` only if profiling shows NetworkX overhead matters.
- Use `mpi4py` only after one-node parallelism is insufficient.
- Do not introduce DOLFINx/FEniCSx unless a genuine finite-element problem emerges.
- Do not use JAX to manage adaptive roots, mesh connectivity, or graph traversal.

### 19.4 `pyproject.toml`

Keep the current minimal base dependencies if desired, and add optional groups such as:

```toml
[project.optional-dependencies]
connectivity = [
    "pyvista",
    "gmsh",
    "networkx",
    "numba",
    "h5py",
    "joblib",
]
mesh-io = ["meshio"]
test = ["pytest", "hypothesis"]
```

Exact minimum versions should be set only after the implementation is tested in the project environment.

---

## 20. Testing strategy

The hardest failures are topological. Tests must go beyond numerical spot checks.
The lists below are the menu of what is worth testing, not a coverage quota; §22.5 sets
the wall-clock budget that any selection from them has to fit inside.

### 20.1 Synthetic fields

Implement analytic `BoozerFieldLike` test fields with controllable:

- one simple trapped well;
- several independent wells;
- a well spanning multiple field periods;
- a generic split/merge controlled by \(s\) or \(\theta\);
- a closed constant-action island disconnected from the edge;
- an edge-connected band;
- a near-tangent separatrix;
- a periodic-seam crossing;
- a field with sine modes to test asymmetry;
- a deliberately degenerate equal-height event.

Synthetic fields should provide exact analytic derivatives.

### 20.2 Unit tests

#### Field evaluation

- Fourier derivatives agree with high-accuracy finite differences.
- Periodicity in \(\theta\) and one field period in \(\zeta\).
- Cosine-only behavior matches the current code.
- Sine modes are loaded and evaluated correctly.
- \(D_\parallel B\) agrees with differentiation along a field line.

#### Coordinates and measure

- \(ds\wedge d\theta=2dx\wedge dy\).
- The \((x,y,\zeta)\) formula for \(\omega\) agrees with a locally unwrapped \((s,\alpha)\) determinant away from the axis.
- The measure remains finite at the axis.

#### Background mesh

- Positive tetrahedron orientation.
- Periodic seam maps are one-to-one.
- No cracks after seam identification.
- Deterministic structured mesh.

#### Surface extraction

- Exact plane cuts through a tetrahedron.
- Closed analytic isosurfaces have expected component counts.
- Extracted vertices satisfy \(|B-b|\) tolerance.
- Incoming and outgoing halves reconstruct the full surface except for \(g=0\).

#### Well trace

For every regular trace:

\[
|B(q_-)-b|<\epsilon,
\qquad
|B(q_+)-b|<\epsilon,
\]

with correct signs of \(g\), and

\[
B<b
\]

between endpoints except within tolerance.

Also test:

- invariance under shifting by one field period;
- long-well period count;
- endpoint quadrature convergence;
- comparison with the existing `compute_J_invariant()` for the same selected well;
- failure statuses rather than silent truncation.

#### Transitions

- recovered \(T\) matches the analytic synthetic transition;
- port correspondence preserves the common parameter;
- \(A_W\approx A_1+A_3\) converges under refinement;
- duplicated cut vertices carry distinct action values;
- nongeneric events are flagged.

#### Interval operations

Use Hypothesis to test:

- commutativity and idempotence of union;
- monotonicity;
- no overlapping stored intervals;
- affine image/preimage consistency;
- tolerance stability;
- serialization round trips.

#### Flood fill

Test small hand-constructed meshes with known answers:

- strip connected to edge;
- closed contour island;
- two disconnected components with overlapping action ranges;
- periodic cylinder seam;
- one generic three-port transition;
- a transition cycle;
- contour through a vertex or constant edge.

For bounded finite-atom mode, also test:

- the inner mask is always a subset of the outer mask;
- every sampled point classified reachable by the inner mask is reachable by direct tracing;
- every sampled point classified unreachable by the outer mask is unreachable by direct tracing;
- nested atom refinement grows the inner reachable set, shrinks the outer reachable set, and reduces unresolved weighted area;
- the worklist terminates deterministically for affine transition cycles.

The optional continuous interval mode must agree with direct contour tracing whenever it reports stabilization.

#### Quadrature

- clipped polygon areas against analytic formulas;
- exact integration of constant and linear weights;
- convergence for a logarithmic boundary weight;
- unresolved lower/upper bounds contain a high-resolution reference.

#### Outer integration

- adaptive midpoint on smooth, kinked, and step-like synthetic functions;
- cache reuse;
- deterministic results independent of task ordering.

### 20.3 Integration tests

1. **Legacy agreement:** for wells near \(\zeta=\pi/N_{\mathrm{fp}}\), new `action_length/L_ref` agrees with existing normalized \(J\).
2. **\(\Theta\equiv1\) benchmark:** the surface/pitch quadrature agrees with an independent direct phase-space trapped-fraction integral.
3. **Flux balance:** incoming and outgoing \(B=b\) surfaces carry equal absolute magnetic flux:
   \[
   \int_{\Sigma_b^-}|d\psi\wedge d\alpha|
   =
   \int_{\Sigma_b^+}|d\psi\wedge d\alpha|.
   \]
4. **One period versus full torus:** replicated calculations give the same \(f\).
5. **QI-like synthetic field:** \(f\) is zero or converges to the expected small value.
6. **Known edge-connected synthetic field:** compare with an analytic or independently discretized result.
7. **W7-X reference data:** run a coarse deterministic smoke test, save diagnostics, and verify finite bounds with \(0\le f\le1\).

### 20.4 Property tests particularly worth adding

- renumbering mesh nodes does not change \(Q(b)\);
- rotating \(\theta\) or shifting \(\zeta\) by a period does not change \(f\);
- refining without changing topology does not destroy an already resolved edge connection;
- every regular incoming point maps to exactly one regular outgoing point;
- every transition port has the same number and ordering of common-parameter samples;
- no well is counted twice in the incoming-surface measure.

---

## 21. Error handling and convergence

### 21.1 Status enums

Define explicit enums such as:

```python
class TraceStatus(Enum):
    REGULAR = auto()
    NO_WELL = auto()
    MAX_PERIODS = auto()
    ROOT_FAILURE = auto()
    QUADRATURE_FAILURE = auto()
    TANGENT_OR_TRANSITION = auto()
    AXIS_UNRESOLVED = auto()
    DEGENERATE = auto()
```

Equivalent statuses are needed for surface extraction, transitions, flood fill, and quadrature.

### 21.2 No silent data loss

The following are forbidden:

- replacing a failed trace with zero action or zero weight;
- interpreting a clipped well as passing;
- dropping triangles containing `NaN` without accounting for their measure;
- treating a missing transition as no connection;
- capping a long trace and assigning \(\Theta=0\);
- merging disconnected surface components because they are close in Euclidean coordinates.

### 21.3 Convergence dimensions

A production result must report convergence or an uncertainty bound with respect to:

1. Fourier/radial field interpolation;
2. background mesh resolution;
3. surface extraction and projection;
4. surface refinement;
5. root scan resolution;
6. root tolerance;
7. \(A\) and \(K\) quadrature tolerances;
8. maximum field-period count;
9. transition-curve sampling;
10. interval merge tolerance or bitmask resolution;
11. surface quadrature;
12. outer \(b\) quadrature;
13. any excluded core.

The result should include a machine-readable `ConvergenceReport` and a human-readable summary.

---

## 22. Parallelism and performance

### 22.1 Expected dominant cost

The likely dominant work is repeated evaluation of \(B\) during:

- outgoing-bounce searches;
- extrema searches;
- \(A\) and \(K\) quadrature;
- transition traces;
- adaptive interior quadrature near marginal curves.

Graph traversal and interval union are expected to be secondary until meshes become very large.

### 22.2 Parallel strategy

Use coarse-grained parallelism over \(b\) slices with `joblib`:

```python
results = Parallel(n_jobs=config.pitch.n_jobs)(
    delayed(compute_pitch_slice)(field_data, background, b, source, config.slice)
    for b in requested_b_values
)
```

Adaptive outer quadrature should submit newly required midpoint slices in batches. Background arrays should be read-only and memory-mapped when process-based workers are used.

Do not allow workers to write concurrently to one HDF5 file.

### 22.3 Numba strategy

Begin with transparent NumPy/SciPy code. After correctness:

- compile field-line scan loops;
- compile Fourier evaluation for fixed radial coefficients;
- compile fixed-node quadrature and extrema bookkeeping;
- compile polygon clipping and cellwise weights if profiling justifies it.

Keep SciPy `brentq` in Python initially. Replace it only if root solving is a measured bottleneck.

### 22.4 Performance instrumentation

Record timings for:

- field loading;
- background mesh;
- surface extraction;
- well traces;
- transition construction;
- mesh cutting;
- flood fill;
- surface quadrature;
- HDF5 I/O;
- outer adaptivity.

Save counts such as field evaluations, roots, trace periods, triangles, transitions, and interval operations.

---

### 22.5 Test-time budget

Test speed is a design constraint, not an afterthought: an agent that cannot run the
suite in a coffee break stops running it. Two tiers, both measured on the researcher's
laptop:

| Tier | Command | Selection | Budget |
| --- | --- | --- | --- |
| fast | `make test` | `-m "not slow"` | under 2 minutes total; no single test over about 20 s |
| full | `make test-full` | everything | under 5 minutes total; no single `slow` test over about 90 s |

The fast tier is what CI runs on every pull request and what an agent runs in its
inner loop. The full tier is the gate before a milestone pull request is marked ready.

Rules:

- The default is fast. A test earns `@pytest.mark.slow` only when the physics it
  covers genuinely cannot be exercised more cheaply, and the milestone pull request
  must say why.
- Before marking a test `slow`, make it cheaper: lower the mesh or Fourier resolution,
  shrink the pitch grid, share expensive fixtures across tests at module scope, cache
  a loaded `boozmn` field.
- Marking a *failing* test `slow`, `xfail`, or `skip` to reach green is never
  acceptable. See §21.2.
- A `slow` test must not be the only live evidence for an acceptance criterion. Every
  scientific claim keeps at least one fast test that goes through the production code
  path and fails under a mutation of the physics it checks. A fast test that only
  checks a schema, an array shape, or that nothing raised does not satisfy this.
- `make test` prints a durations report. It is the early-warning system; when the
  slowest fast test starts approaching 20 s, fix it in that pull request rather than
  leaving it for the next one.

This repository does not aim for exhaustive coverage of every parameter combination.
It aims for a small suite of tests that would actually catch a wrong sign, a dropped
term, or a misclassified topology, and that runs fast enough to be run every time.

---

## 23. AI-agent milestones

Each milestone should normally be one focused pull request. An agent must read this document and `AGENTS.md`, run the full existing test suite before and after its changes, and add both tests and at least one diagnostic example for any geometric feature.

Completion state lives in `docs/STATUS.md`, not here. This section defines what each milestone *is*; `docs/STATUS.md` records which ones are done. A milestone's implementing pull request marks its own row there. Milestones are ordered: unless the researcher says otherwise, take the lowest-numbered unchecked row.

### Milestone 0: Baseline and design scaffolding

**Goal:** Establish package skeleton without changing numerical behavior.

**Changes:**

- add `alpha_analysis/j_connectivity/` with `__init__.py`, `config.py`, and `types.py`;
- add optional dependency groups to `pyproject.toml`;
- add a trivial import test;
- add run metadata and status enums;
- preserve all current CLIs and tests.

**Acceptance:** Existing tests pass; new package imports with base dependencies;
optional imports fail with informative messages.

### Milestone 1: General Boozer field derivatives and asymmetric modes

**Goal:** Provide the field interface needed by all later work.

**Changes:**

- extend `BoozerField` or add an adapter for sine coefficients;
- implement analytic \(s\), \(\theta\), \(\zeta\), \(D_\parallel\), and \(D_\parallel^2\) derivatives;
- expose \(C=G+\iota I\);
- add synthetic Fourier fields;
- add finite-difference and periodicity tests.

**Acceptance:** Derivative tests pass for cosine and sine modes; current `compute_B` behavior is unchanged for existing data.

### Milestone 2: Denominator \(V_h\) and global \(B\) bounds

**Goal:** Implement the independent normalization calculation.

**Changes:**

- tensor-product quadrature for \(V_h\);
- source-profile API;
- refined global \(B\) extrema search;
- convergence diagnostics and plots of \(B_{\min}(s)\), \(B_{\max}(s)\), and denominator convergence.

**Acceptance:** Manufactured-field integrals agree with analytic/reference
results; periodic resolution convergence is demonstrated.

### Milestone 3: Deterministic periodic background mesh

**Goal:** Build the axis-regular logical mesh without Gmsh.

**Changes:**

- disk triangulation, extrusion, prism-to-tet split;
- periodic seam map;
- boundary tags;
- PyVista conversion and mesh plots;
- mesh quality tests.

**Acceptance:** No inverted tetrahedra; exact seam pairing; deterministic output; visual example generated.

### Milestone 4: Gmsh background backend

**Goal:** Add the production mesher behind the same interface.

**Changes:**

- logical-cylinder geometry;
- periodic end-surface relation;
- extraction of node, tetrahedron, and periodic-pair arrays;
- optional size fields;
- comparison with structured backend.

**Acceptance:** Mesh passes the same invariants; Gmsh is isolated from core numerical data structures.

### Milestone 5: \(B=b\) surface extraction

**Goal:** Extract all level-surface components and incoming/outgoing halves.

**Changes:**

- PyVista prototype;
- custom marching-tetrahedra reference or production path;
- edge-root polishing;
- periodic merge;
- split by \(g=0\);
- boundary tags and 3D visualizations.

**Acceptance:** Synthetic isosurfaces have correct topology; vertex residuals meet tolerance; incoming/outgoing flux balance begins to converge.

### Milestone 6: Regular well tracer

**Goal:** Trace every regular surface vertex, not only the well near \(\pi/N_{\mathrm{fp}}\).

**Changes:**

- unwrapped field-line scan;
- outgoing root and extrema detection;
- endpoint-regularized \(A\) and \(K\);
- itinerary records and statuses;
- well-profile plots;
- comparison against existing `compute_J_invariant()`.

**Acceptance:** All regular synthetic tests pass; legacy selected-well agreement is within tolerance; long wells preserve period count.

### Milestone 7: Surface data, refinement, and sheet candidates

**Goal:** Evaluate action data over a whole pitch surface and refine discontinuity candidates.

**Changes:**

- batch traces at vertices;
- interpolation-error estimators;
- itinerary comparison;
- local surface refinement and projection;
- visual maps of \(A\), \(K\), exit, maxima count, and statuses.

**Acceptance:** Smooth manufactured action fields converge; candidate return-map discontinuities sharpen with refinement.

### Milestone 8: Critical curves

**Goal:** Robustly extract and classify \(\Gamma_{\min}\), \(\Gamma_{\max}\), and degenerate portions.

**Changes:**

- polyline connectivity and periodic stitching;
- second-derivative classification;
- refinement around ambiguous segments;
- curve plots.

**Acceptance:** Synthetic critical curves match analytic locations and classifications.

### Milestone 9: Transition mapping and action additivity

**Goal:** Construct \(T\) and matched parent/child ports without yet cutting the full mesh.

**Changes:**

- backward and forward tangent-event traces;
- common parameter samples;
- \(A_W,A_1,A_3\);
- additivity diagnostics;
- multiway-event detection;
- transition plots.

**Acceptance:** Generic synthetic split satisfies additivity under refinement; mismatched nearest-neighbor associations are impossible because lifted field-line identity is retained.

### Milestone 10: Constrained cuts and sheet IDs

**Goal:** Insert \(T\), duplicate vertices, and make \(A\) continuous on each sheet.

**Changes:**

- polyline insertion into triangular mesh;
- vertex/edge duplication;
- branch-specific action values;
- union-find sheet IDs;
- coarse NetworkX transition graph;
- before/after cut plots.

**Acceptance:** No triangle spans an action jump; each port has a valid sheet; topology survives serialization.

### Milestone 10.1: Sampling-robust cut geometry

Inserted 2026-08-30 after the milestone-10 real-equilibrium matrix resolved no cut:
the transition-mapping sample subset (`max_curve_samples`) directly becomes the
inserted cut polyline, so the sample budget silently changes cut geometry and, on
under-sampled curves, whether the cut resolves at all — the §21.3 dimension-9
fragility ADR 0005 documented.

**Goal:** Make the sheet graph invariant to the transition-mapping sample budget, or explicitly budget-limited — never silently budget-dependent.

**Changes:**

- `max_curve_samples` becomes a work budget: the sample subset densifies adaptively from the authoritative critical-curve vertices where a mapped midpoint disagrees with interpolation from its neighbors (geometric deviation, action deviation, itinerary change, EDGE proximity, near self-contact);
- an explicit budget-insufficient transition status/reason when certification is not reached within budget, instead of cutting a different, coarser curve;
- budget-invariance acceptance checks: identical sheet graphs across budgets (8, 10, 16, full) or an explicit budget report;
- documentation no longer describes the subset as "bounding cost without coarsening geometry".

**Acceptance:** A named test demonstrates sheet-graph invariance across sample budgets on the production synthetic field and on the DMercFail reference equilibrium, and a named test demonstrates the explicit budget-insufficient path; no cut resolves or changes topology as a silent function of the budget.

### Milestone 10.2: Contact localization and segment-level cutting

Inserted 2026-08-30: after ADR 0003, 57 of the 164 real-matrix transition curves are
`MULTIWAY` because coarse sampling steps over an interior-maximum count change
somewhere along the curve — only 6 of 164 are free of one — and the all-or-nothing
resolvability gate then vetoes the entire curve, so real equilibria essentially never
cut at any affordable sampling.

**Goal:** Cut the resolved arcs of a transition curve whose nongeneric events are localized, keeping every event an explicit §5.4 hyperedge.

**Changes:**

- bisection in \(u\) of ADR 0003 contact brackets (each bracketed event is localized to a tight interval or dissolved as a sampling artifact by a few extra traces);
- subdivision of transition curves at localized events and at nongeneric samples;
- explicit event nodes with arbitrary port count (§5.4) at subdivision points, including the sheet-graph treatment of a cut terminating at an interior event junction;
- per-arc resolvability and cutting, replacing the whole-curve gate, with the nongeneric arcs retained as explicit unresolved or event hyperedges.

**Acceptance:** A curve with one bracketed contact cuts its regular arcs and carries an explicit event hyperedge at the localized contact — never an arbitrary binary decomposition; the W7-X reference curve with four contacts yields cut arcs plus explicit events; no dangling cut terminates in a surface interior without an event node.

### Milestone 10.3: Failure-directed refinement and matrix convergence

Inserted 2026-08-30: the remaining real-matrix failures are resolution effects with
machine-readable reasons (unresolved surfaces and critical curves, thin
\(T\)-to-`EDGE` strips, off-component projections, `MAX_PERIODS` caps), each with a
demonstrated targeted remedy from the milestone-9/10 convergence probes — but today
each remedy requires manual per-case tuning.

**Goal:** Make the five-equilibrium matrix converge unattended: every failure class triggers its targeted remediation, bounded, and what remains unresolved is physics or an explicit budget, not a default knob.

**Changes:**

- a coordinator that dispatches on the recorded unresolved reason: background refinement for unresolved extractions/critical curves, \(u\)-refinement for contact brackets, per-sample period-cap escalation for `MAX_PERIODS`, component-provenance enforcement for off-surface projections, and local surface refinement near thin transition strips;
- local (not global) surface refinement around companion curves whose strip width fails the resolution requirement;
- bounded escalation with every retry recorded (§21.3);
- regeneration of the real-equilibrium validation matrix with budget-invariance checks.

**Acceptance:** The 100-case matrix report shows every case either resolves or terminates with a physically meaningful reason (a genuinely unrepresentable strip at the refinement bound, a genuine cap ceiling) with counts per failure class; no case needs manual per-case tuning to resolve.

### Milestone 11: Direct contour tracer

**Goal:** Build the correctness oracle.

**Changes:**

- PL contour traversal;
- periodic adjacency;
- transition branching;
- loop detection and memoization;
- interactive contour plots.

**Acceptance:** Hand-built and synthetic reachability cases are correct; no infinite loops; critical conventions are tested.

### Milestone 12: Interval primitives and bounded ordinary flood fill

**Goal:** Classify edge-connected action ranges without transitions using a finite algorithm with lower and upper bounds.

**Changes:**

- `IntervalSet` and serialization for exact local operations;
- finite action-atom grids and sparse bitmasks;
- inner/outward snapping rules;
- triangle work queues, edge seeds, and ordinary propagation;
- property-based tests;
- visualization of lower, upper, and unresolved per-triangle intervals.

**Acceptance:** The worklist terminates deterministically; lower and upper results bracket direct tracing on random regular query points; the unresolved weighted area converges to zero under atom refinement for surfaces without transitions.

### Milestone 13: Transition-aware bounded flood fill

**Goal:** Add common-parameter transfer through hyperedges while preserving finite termination and bounds.

**Changes:**

- finite transition-parameter atom grids;
- affine preimage/image mappings with inner and outer snapping;
- monotone segment subdivision;
- optional continuous-interval accelerator with iteration cap;
- comparison with direct tracer.

**Acceptance:** Generic splits, affine transition cycles, and periodic cases terminate; lower/upper classifications bracket direct tracing and converge under joint action/parameter refinement.

### Milestone 14: Reachable-polygon and surface quadrature

**Goal:** Compute \(Q(b)\).

**Changes:**

- triangle clipping by action intervals;
- regular weighted quadrature;
- singular boundary-triangle treatment;
- per-sheet/radial contributions;
- lower/upper unresolved bounds;
- polygon and contribution plots.

**Acceptance:** Analytic polygon tests and logarithmic convergence tests pass; \(Q(b)\) is invariant to triangle ordering.

### Milestone 15: Pitch-slice pipeline and HDF5

**Goal:** Produce a restartable `PitchSliceResult` end to end.

**Changes:**

- orchestration;
- HDF5 schema;
- cache keys;
- VTK bundle output;
- human-readable pitch-slice report.

**Acceptance:** Interrupted runs restart without recomputing completed stages; round-trip data preserve topology and intervals.

### Milestone 16: Adaptive outer quadrature and parallel execution

**Goal:** Compute final \(f\).

**Changes:**

- adaptive midpoint tree;
- joblib batching;
- parent-only HDF5 merge;
- `FractionResult` and convergence plots.

**Acceptance:** Manufactured integrands pass; deterministic result independent of worker count and task completion order.

### Milestone 17: Full validation suite

**Goal:** Establish scientific credibility on synthetic and repository data.

**Changes:**

- \(\Theta\equiv1\) benchmark;
- flux-balance check;
- one-period/full-torus check;
- W7-X coarse integration test;
- convergence dashboard;
- documented known limitations.

**Acceptance:** All required validations pass or produce explicit quantified unresolved bounds.

### Milestone 18: Profiling and Numba acceleration

**Goal:** Optimize only measured bottlenecks.

**Changes:**

- benchmark suite;
- Numba field-line scan and quadrature kernels;
- memory profiling;
- performance report.

**Acceptance:** Numerical results remain unchanged within tolerance; meaningful speedup is demonstrated on a representative pitch slice.

---

## 24. Agent implementation protocol

For every milestone, the implementing agent should:

1. read this design and `AGENTS.md`;
2. identify the exact acceptance tests before coding;
3. avoid broad refactors unrelated to the package;
4. preserve existing public behavior;
5. use plain arrays in the numerical core;
6. add docstrings defining conventions and units;
7. add at least one visualization or diagnostic for new geometry;
8. add unit tests and run all existing tests;
9. record any deviation from this design in the pull-request description;
10. never hide an unresolved topology or numerical failure.

A reviewer should reject a pull request that produces plausible pictures but lacks machine-checkable invariants.

---

## 25. Scientific validation checklist

Before a result is described as converged, verify:

- [ ] all surface vertices satisfy the \(B=b\) residual tolerance;
- [ ] incoming vertices have the correct physical sign of \(\mathbf b\cdot\nabla B\);
- [ ] every regular trace has one first outgoing root and stays below \(b\) in between;
- [ ] periodic shifts leave \(A\), \(K\), itinerary, and \(\Theta\) unchanged;
- [ ] all identified generic transitions satisfy action additivity;
- [ ] every itinerary discontinuity is explained by a transition or marked unresolved;
- [ ] no triangle crosses an uncut action discontinuity;
- [ ] direct contour tracing and flood fill agree away from critical values;
- [ ] incoming/outgoing surface fluxes agree under refinement;
- [ ] the \(\Theta\equiv1\) benchmark agrees with an independent integral;
- [ ] the denominator is independently converged;
- [ ] the outer \(b\) integral is converged;
- [ ] maximum trace-period and omitted-core bounds are negligible or reported;
- [ ] \(0\le f_{\mathrm{lower}}\le f\le f_{\mathrm{upper}}\le1\);
- [ ] results are reproducible after HDF5 restart;
- [ ] all diagnostic plots identify unresolved data visibly.

---

## 26. Design decisions and rationale

### Decision 1: Mesh incoming bounce points, not \((\rho,\alpha)\)

The global \((\rho,\alpha)\) projection is multivalued and develops folds when wells split or merge. The physical incoming surface \(\Sigma_b^-\) is the correct two-dimensional state space and counts each well once.

### Decision 2: Process fixed \(B_b\) slices

\(B_b=W_0/\mu\) is conserved. Independent surfaces allow pitch-specific adaptivity and parallelism. A shared volume mesh retains reuse without forcing every pitch to share the same refinement.

### Decision 3: Use action contours, not a bounce-averaged drift ODE

Once \(A\) is tabulated, drift trajectories are its level contours. Piecewise-linear contour topology avoids repeated bounce-average evaluation inside an ODE callback.

### Decision 4: Use explicit transition hyperedges

A split/merge changes the action discontinuously. It cannot be represented by ordinary interpolation across a triangle. The mesh must be cut and the permitted branch relation stored explicitly.

### Decision 5: Use bounded interval/atom flood fill for production

The scalar integral requires classifying a continuum of states. Propagating reachable action ranges amortizes the work over all contours and maps directly to PL polygon clipping. The production path uses finite action and transition-parameter atoms with inner and outer snapping, so it terminates and supplies lower/upper bounds. Continuous interval propagation is an optional accelerator, not the sole correctness argument.

### Decision 6: Retain direct contour tracing

The flood fill is efficient but globally sensitive to topology errors. A transparent pointwise tracer is the most valuable correctness oracle.

### Decision 7: Use axis-regular \((x,y,\zeta)\) geometry

Polar logical coordinates collapse at \(\rho=0\). The Cartesianized disk gives a valid tetrahedral mesh and a regular expression for the phase-space two-form.

### Decision 8: Keep external libraries outside the scientific topology core

Gmsh, PyVista, and NetworkX provide mature generic operations, but none knows the first-return well identity or the physical transition rule. Those remain explicit custom code over NumPy arrays.

---

## 27. Future extensions

After version 1 is validated, possible extensions include:

1. **Probabilistic transitions:** replace Boolean hyperedges with branch probabilities based on separatrix-crossing theory and solve an absorbing Markov problem.
2. **Directed dynamics:** retain the orientation of the bounce-averaged drift rather than treating contours as undirected.
3. **Reeb graph acceleration:** build an explicit Reeb graph on each cut sheet and compare with the bounded flood fill.
4. **TTK validation:** use Topology ToolKit on continuous sheet fields after cuts.
5. **MPI:** distribute pitch slices over nodes.
6. **Continuation in \(b\):** deform one extracted surface into neighboring pitch values and reuse triangulations and root brackets.
7. **Optimization support:** differentiate a regularized or probabilistic version of the metric.
8. **Finite-orbit-width comparison:** compare the topological accessibility bound with guiding-center orbit following.

---

## 28. Definition of done for version 1

Version 1 is complete when the repository can, from a Boozer equilibrium and a source profile:

1. construct all fixed-\(B_b\) incoming-bounce surfaces needed by an adaptive outer quadrature;
2. enumerate all regular wells on those surfaces rather than only a preferred well near \(\zeta=\pi/N_{\mathrm{fp}}\);
3. compute \(A\), \(K\), exit winding, extrema, and itinerary data;
4. construct and validate generic split/merge transitions;
5. cut the surface so \(A\) is continuous on each sheet;
6. compute edge reachability with both direct contour tracing and bounded action/transition-atom flood fill;
7. integrate the reachable phase-space weight and form \(f\);
8. report lower/upper bounds for unresolved regions;
9. checkpoint and restart the calculation;
10. generate the required mesh, surface, transition, reachability, quadrature, and convergence visualizations;
11. pass the synthetic, legacy-regression, flux-balance, \(\Theta\equiv1\), periodicity, and W7-X smoke tests;
12. provide a convergence report sufficient to judge whether a quoted value of \(f\) is trustworthy.
