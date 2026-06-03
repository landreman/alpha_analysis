import os

import matplotlib.pyplot as plt
import numpy as np
import alpha_analysis.J_invariant as J_invariant_module

from alpha_analysis import (
    DATA_DIR,
    BoozerField,
    BoozerSurface,
    compute_J_invariant,
    find_bounce_points,
)

boozmn_file_name = os.path.join(
    DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
)


def test_compute_J_invariant_reference_grid():
    booz = BoozerField.from_boozmn(boozmn_file_name)

    alpha_values = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    rho_values = np.array([0.5, 1.0])
    s_values = rho_values**2
    lambda_n_values = np.array([0.1, 0.3, 0.5, 0.8])

    b_min, b_max = booz.get_min_max()
    phi_center = np.pi / booz.nfp

    j_reference = np.array(
        [
            [
                [0.00873977512318135, 0.023394976504410036],
                [0.043955587975138856, 0.02417013078841274],
                [0.047411593347227256, 0.021248913705857228],
                [0.023917904151112786, 0.024170130788412694],
            ],
            [
                [0.14376622898855615, 0.091361876815906],
                [0.1488120349311294, 0.11350986852550063],
                [0.15033078696630559, 0.11476743295690289],
                [0.14665286628320484, 0.0873463502918537],
            ],
            [
                [0.5668853990400646, 0.4598254566459877],
                [0.253930388305994, 0.1987444796980752],
                [0.25089923422733934, 0.20052247249992536],
                [0.8453458338883063, 0.4396318158857731],
            ],
            [
                [np.nan, 1.4271735115821336],
                [np.nan, 1.4192534097688618],
                [np.nan, 0.3250939627968473],
                [np.nan, 1.4192534097688614],
            ],
        ]
    )

    j_computed = np.empty_like(j_reference)

    for lambda_idx, lambda_n in enumerate(lambda_n_values):
        b_bounce = b_min + lambda_n * (b_max - b_min)

        for rho_idx, s in enumerate(s_values):
            surf = BoozerSurface(booz, s)
            for alpha_idx, alpha in enumerate(alpha_values):
                theta_center = alpha + surf.iota * phi_center
                data = compute_J_invariant(
                    surf,
                    b_bounce,
                    theta_center,
                    phi_center,
                    n_phi=1001,
                    phi_margin=5.0,
                    refine=True,
                )
                j_computed[lambda_idx, alpha_idx, rho_idx] = data["J"]

    np.testing.assert_allclose(
        j_computed,
        j_reference,
        rtol=1e-13,
        atol=1e-13,
        equal_nan=True,
    )

def test_J_refine_doesnt_change_too_much():
    """The J invariant computed with refine=True should be close to that found with refine=False."""
    booz = BoozerField.from_boozmn(boozmn_file_name)
    s = 0.5
    surf = BoozerSurface(booz, s)

    B_bounces = [0.1, 2.4, 2.7, 3.1, 5.1]
    alphas = np.linspace(0.0, 2.0 * np.pi, 10)
    phi_center = np.pi / surf.nfp
    n_phi = 1001
    phi_margin = 4.0

    for B_bounce in B_bounces:
        for alpha in alphas:
            data_refined = compute_J_invariant(
                surf,
                B_bounce,
                alpha,
                phi_center,
                n_phi=n_phi,
                phi_margin=phi_margin,
                refine=True,
            )

            data_unrefined = compute_J_invariant(
                surf,
                B_bounce,
                alpha,
                phi_center,
                n_phi=n_phi,
                phi_margin=phi_margin,
                refine=False,
            )

            np.testing.assert_allclose(
                data_refined["J"], data_unrefined["J"], atol=1e-14, rtol=0.004
            )


def test_compute_single_j_grid_refine_matches_pointwise_reference():
    booz = BoozerField.from_boozmn(boozmn_file_name)
    alpha_values = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    s_values = np.array([0.25, 0.5])
    lambda_n = 0.35
    phi_center = np.pi / booz.nfp
    b_min, b_max = booz.get_min_max()
    b_bounce = b_min + lambda_n * (b_max - b_min)

    j_grid = J_invariant_module._compute_single_j_grid(
        booz,
        alpha_values,
        s_values,
        lambda_n=lambda_n,
        refine=True,
    )

    for s_idx, s in enumerate(s_values):
        surf = BoozerSurface(booz, s)
        for a_idx, alpha in enumerate(alpha_values):
            theta_center = alpha + surf.iota * phi_center
            data = compute_J_invariant(
                surf,
                b_bounce,
                theta_center,
                phi_center,
                refine=True,
            )
            np.testing.assert_allclose(
                j_grid[a_idx, s_idx],
                data["J"],
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
            )


def test_plot_J_invariant_cli_forwards_args(monkeypatch):
    captured = {}

    def _fake_plot_J_invariant(
        boozmn_file,
        n_alpha,
        n_rho,
        contour_levels,
        refine,
        show=True,
    ):
        captured["boozmn_file"] = boozmn_file
        captured["n_alpha"] = n_alpha
        captured["n_rho"] = n_rho
        captured["contour_levels"] = contour_levels
        captured["refine"] = refine
        captured["show"] = show

    monkeypatch.setattr(J_invariant_module, "plot_J_invariant", _fake_plot_J_invariant)
    exit_code = J_invariant_module.plot_J_invariant_cli(
        [
            boozmn_file_name,
            "--n_alpha",
            "12",
            "--n_rho",
            "15",
            "--contour_levels",
            "27",
            "--no-refine",
        ]
    )

    assert exit_code == 0
    assert captured["boozmn_file"] == boozmn_file_name
    assert captured["n_alpha"] == 12
    assert captured["n_rho"] == 15
    assert captured["contour_levels"] == 27
    assert captured["refine"] is False
    assert captured["show"] is True


def test_plot_J_invariant_single_lambda_cli_forwards_args(monkeypatch):
    captured = {}

    def _fake_plot_J_invariant_single_lambda(
        boozmn_file,
        lambda_n,
        n_alpha,
        n_rho,
        contour_levels,
        refine,
        show=True,
    ):
        captured["boozmn_file"] = boozmn_file
        captured["lambda_n"] = lambda_n
        captured["n_alpha"] = n_alpha
        captured["n_rho"] = n_rho
        captured["contour_levels"] = contour_levels
        captured["refine"] = refine
        captured["show"] = show

    monkeypatch.setattr(
        J_invariant_module,
        "plot_J_invariant_single_lambda",
        _fake_plot_J_invariant_single_lambda,
    )
    exit_code = J_invariant_module.plot_J_invariant_single_lambda_cli(
        [
            boozmn_file_name,
            "0.35",
            "--n_alpha",
            "12",
            "--n_rho",
            "15",
            "--contour_levels",
            "27",
            "--refine",
        ]
    )

    assert exit_code == 0
    assert captured["boozmn_file"] == boozmn_file_name
    assert captured["lambda_n"] == 0.35
    assert captured["n_alpha"] == 12
    assert captured["n_rho"] == 15
    assert captured["contour_levels"] == 27
    assert captured["refine"] is True
    assert captured["show"] is True


def test_save_combined_pdf_writes_output(tmp_path):
    fig1, ax1 = plt.subplots()
    ax1.plot([0, 1], [0, 1])
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [1, 0])

    output_path = tmp_path / "combined.pdf"
    J_invariant_module._save_combined_pdf([fig1, fig2], output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig1)
    plt.close(fig2)


def test_get_subplot_contour_data_uses_requested_linear_range():
    j_plot = np.array([[1.0, 2.0], [3.0, 4.0]])

    plot_data, levels, norm = J_invariant_module._get_subplot_contour_data(
        j_plot,
        "linear",
        5,
        vmin=1.5,
        vmax=3.5,
    )

    np.testing.assert_array_equal(plot_data, j_plot)
    np.testing.assert_allclose(levels, np.linspace(1.5, 3.5, 5))
    assert norm is None


def test_plot_single_lambda_gui_keeps_widgets_and_uses_cartesian_axis_labels():
    alpha_values = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    rho_values = np.array([0.2, 0.6, 1.0])
    j_grid = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [2.0, 3.0, 4.0],
            [2.5, 3.5, 4.5],
        ]
    )

    fig = J_invariant_module._plot_single_lambda_gui(
        alpha_values,
        rho_values,
        j_grid,
        contour_levels=7,
        boozmn_path=os.path.abspath(boozmn_file_name),
        lambda_n=0.35,
        n_alpha=4,
        n_rho=3,
        refine=False,
    )

    assert hasattr(fig, "_single_lambda_widgets")
    assert fig._single_lambda_widgets["contour_slider"].val == 7
    assert fig._single_lambda_widgets["filled_toggle"].get_status() == [True]
    assert fig.axes[0].get_title() == ""
    assert "linear color scale" not in fig._suptitle.get_text()
    assert r"\lambda_n=0.35" in fig._suptitle.get_text()
    assert not fig.axes[0].xaxis.get_gridlines()[0].get_visible()
    assert fig.axes[0].get_xlabel() == r"$x=\rho\cos\alpha$"
    assert fig.axes[0].get_ylabel() == r"$y=\rho\sin\alpha$"
    assert any(label.get_text() != "" for label in fig.axes[0].get_xticklabels())
    assert any(label.get_text() != "" for label in fig.axes[0].get_yticklabels())

    J_invariant_module.plt.close(fig)


def test_plot_large_polar_figures_appends_b_extrema_subplot(tmp_path):
    alpha_values = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    rho_values = np.array([0.2, 0.6, 1.0])
    lambda_n_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    base_grid = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [2.0, 3.0, 4.0],
            [2.5, 3.5, 4.5],
        ]
    )
    j_grids = {lambda_n: base_grid + lambda_n for lambda_n in lambda_n_values}
    b_extrema = {
        "min": np.array([0.8, 0.7, 0.6]),
        "max": np.array([1.8, 1.9, 2.0]),
    }

    figures = J_invariant_module._plot_large_polar_figures(
        alpha_values,
        rho_values,
        j_grids,
        lambda_n_values,
        contour_levels=6,
        output_base_path=tmp_path / "large.pdf",
        boozmn_path=os.path.abspath(boozmn_file_name),
        n_alpha=len(alpha_values),
        n_rho=len(rho_values),
        refine=False,
        b_extrema=b_extrema,
        b_min=0.5,
        b_max=2.5,
        n_rows=2,
        n_cols=3,
    )

    assert len(figures) == 1
    b_axes = [ax for ax in figures[0].axes if ax.get_ylabel() == r"$B$"]
    assert len(b_axes) == 1
    assert b_axes[0].get_xlabel() == r"$\rho$"
    assert b_axes[0].get_title() == r"$B$ extrema"
    assert len(b_axes[0].collections) == len(lambda_n_values)
    assert len(b_axes[0].lines) == 2
    expected_colors = [
        plt.get_cmap("jet")(
            (lambda_n - min(lambda_n_values)) / (max(lambda_n_values) - min(lambda_n_values))
        )
        for lambda_n in reversed(lambda_n_values)
    ]
    actual_colors = [collection.get_colors()[0] for collection in b_axes[0].collections]
    for actual_color, expected_color in zip(actual_colors, expected_colors):
        np.testing.assert_allclose(actual_color, expected_color)
    legend = b_axes[0].get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels[:2] == [r"$\max(B)$", r"$\min(B)$"]
    assert rf"$\lambda_n={lambda_n_values[-1]:.2f}$" in legend_labels

    J_invariant_module.plt.close(figures[0])


def test_compute_j_grids_refine_false_reuses_B_evaluations(monkeypatch):
    booz = BoozerField.from_boozmn(boozmn_file_name)
    alpha_values = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    s_values = np.array([0.2, 0.6])

    count = 0
    original_compute_B = BoozerSurface.compute_B_tensor_alpha_phi

    def _counted_compute_B(self, alpha, phi):
        nonlocal count
        count += 1
        return original_compute_B(self, alpha, phi)

    monkeypatch.setattr(BoozerSurface, "compute_B_tensor_alpha_phi", _counted_compute_B)
    J_invariant_module._compute_j_grids(booz, alpha_values, s_values, refine=False)

    assert count == len(s_values)


def test_cached_unrefined_j_matches_public_unrefined_paths():
    booz = BoozerField.from_boozmn(boozmn_file_name)
    surf = BoozerSurface(booz, s=0.5)

    phi_center = np.pi / surf.nfp
    n_phi = 501
    phi_margin = 5.0
    phi_field_period = 2.0 * np.pi / surf.nfp
    phi = (
        phi_center
        + np.linspace(-phi_margin - 0.5, phi_margin + 0.5, n_phi) * phi_field_period
    )

    b_bounces = [2.4, 2.7, 5.1]
    alphas = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)

    for b_bounce in b_bounces:
        for alpha in alphas:
            theta_center = alpha + surf.iota * phi_center
            theta = theta_center + surf.iota * (phi - phi_center)
            B = surf.compute_B(theta, phi)

            cached_data = J_invariant_module._compute_unrefined_j_from_cached_B(
                surf=surf,
                B=B,
                phi=phi,
                B_bounce=b_bounce,
                clipped_well_nan=True,
                return_data=True,
            )

            bounce_data = find_bounce_points(
                surf,
                b_bounce,
                theta_center,
                phi_center,
                n_phi=n_phi,
                phi_margin=phi_margin,
                refine=False,
            )

            j_data = compute_J_invariant(
                surf,
                b_bounce,
                theta_center,
                phi_center,
                n_phi=n_phi,
                phi_margin=phi_margin,
                refine=False,
                clipped_well_nan=True,
            )

            np.testing.assert_array_equal(cached_data["allowed"], bounce_data["allowed"])
            np.testing.assert_array_equal(cached_data["well_mask"], bounce_data["well_mask"])
            np.testing.assert_equal(
                cached_data["well_crosses_left_edge"],
                bounce_data["well_crosses_left_edge"],
            )
            np.testing.assert_equal(
                cached_data["well_crosses_right_edge"],
                bounce_data["well_crosses_right_edge"],
            )
            np.testing.assert_equal(cached_data["left_index"], bounce_data["left_index"])
            np.testing.assert_equal(cached_data["right_index"], bounce_data["right_index"])
            np.testing.assert_allclose(
                cached_data["J"],
                j_data["J"],
                atol=1e-14,
                rtol=1e-14,
                equal_nan=True,
            )
