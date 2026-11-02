"""HDR float32 + sampling-bug regression tests for equi2pers.

Covers the four defects blocking 24k Poly Haven EXR (float32 radiance up to
~1e4) reprojection:
  1. HDR destruction (uint8 clip/cast)
  2. Bilinear degenerated to nearest (int truncation)
  3. Longitude seam smear (clamp instead of wrap)
  4. Pole wrap (latitude % wrap instead of clamp)
"""
import numpy as np
import pytest

from equiforge.converters.equi2pers import equi2pers, equi2pers_cpu, process_chunk
from equiforge.utils.sampling import bilinear_sampling


def test_float32_constant_hdr():
    """Constant 1000.0 equirect stays ~1000.0 with float32 output."""
    equi = np.full((60, 120, 3), 1000.0, dtype=np.float32)
    out = equi2pers(equi, output_width=32, output_height=32, fov_x=90,
                    use_gpu=False, output_dtype="float32")
    assert out.dtype == np.float32
    assert out.shape == (32, 32, 3)
    assert np.allclose(out, 1000.0, atol=1e-3), f"min={out.min()} max={out.max()}"


def test_default_uint8_clipped():
    """Default call still returns uint8 clipped to [0, 255]."""
    equi_hdr = np.full((60, 120, 3), 1000.0, dtype=np.float32)
    out = equi2pers(equi_hdr, output_width=32, output_height=32,
                    fov_x=90, use_gpu=False)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    # 1000 >> 255, so everything must be clipped to white.
    assert np.all(out == 255)

    # Normal-range uint8 input still round-trips as uint8.
    equi_u8 = np.full((60, 120, 3), 128, dtype=np.uint8)
    out2 = equi2pers(equi_u8, output_width=16, output_height=16,
                     fov_x=90, use_gpu=False)
    assert out2.dtype == np.uint8


def test_invalid_output_dtype():
    with pytest.raises(ValueError):
        equi2pers(np.zeros((8, 16, 3), dtype=np.uint8),
                  output_width=8, output_height=8,
                  use_gpu=False, output_dtype="uint16")


def test_true_bilinear_half_pixel():
    """Bilinear must interpolate, not truncate to nearest.

    1-row ramp: sampling halfway between pixels 0 and 1 must give ~5,
    and the equi2pers centre ray aimed at equi_x=100.5 must give ~100.5
    (the old int() truncation returned 100.0).
    """
    # Direct 1-row ramp unit check.
    img = np.zeros((1, 4, 3), dtype=np.float32)
    img[0, :, 0] = np.array([0, 10, 20, 30], dtype=np.float32)
    v = bilinear_sampling(img, 0.0, 0.5, False)
    assert 4.0 < float(v[0]) < 6.0, f"expected ~5, got {v}"

    # End-to-end through equi2pers: ramp across width, yaw puts centre at x=100.5.
    W, H = 200, 50
    equi = np.zeros((H, W, 3), dtype=np.float32)
    ramp = np.arange(W, dtype=np.float32)
    for c in range(3):
        equi[:, :, c] = np.tile(ramp, (H, 1))
    out = equi2pers(equi, output_width=11, output_height=11, fov_x=90,
                    yaw=0.9, pitch=0.0, roll=0.0,
                    use_gpu=False, output_dtype="float32",
                    sampling_method="bilinear")
    centre = float(out[5, 5, 0])
    assert 100.2 < centre < 100.8, f"expected ~100.5, got {centre}"
    # Sanity: yaw=0 centres exactly on pixel 100.
    out0 = equi2pers(equi, output_width=11, output_height=11, fov_x=90,
                     yaw=0.0, use_gpu=False, output_dtype="float32")
    assert abs(float(out0[5, 5, 0]) - 100.0) < 0.3


def test_seam_no_discontinuity():
    """View centred on yaw=180 (the +/-180 deg seam) must stay smooth.

    Equirect value is a smooth periodic function of longitude (continuous
    across the wrap). With float32 output and a narrow FOV the centre row
    must not show a jump at the seam column; the old clamp-everything code
    smeared the last column instead of blending last<->first.
    """
    W, H = 400, 200
    equi = np.zeros((H, W, 3), dtype=np.float32)
    xs = np.arange(W, dtype=np.float64)
    theta = xs / W * 2 * np.pi - np.pi
    vals = 2000.0 + 1000.0 * np.sin(4 * theta)  # periodic: continuous across wrap
    for j in range(W):
        equi[:, j, :] = np.float32(vals[j])

    out = equi2pers(equi, output_width=64, output_height=64, fov_x=10,
                    yaw=180, pitch=0, roll=0,
                    use_gpu=False, output_dtype="float32")
    assert out.dtype == np.float32
    centre_row = out[32, :, 0].astype(np.float64)
    diffs = np.abs(np.diff(centre_row))
    # Smooth sine gradient is ~11/px here; a seam smear would spike to ~60.
    assert diffs.max() < 30.0, f"seam discontinuity: max diff {diffs.max()}"
    # And the direct sampler must blend last<->first when wrapping.
    img = np.zeros((1, 4, 3), dtype=np.float32)
    img[0, :, 0] = np.array([0, 10, 20, 30], dtype=np.float32)
    wrapped = float(bilinear_sampling(img, 0.0, 3.5, True)[0])
    assert 14.0 < wrapped < 16.0, f"wrap blend expected ~15, got {wrapped}"


def test_pole_does_not_wrap_to_top():
    """Latitude clamps: looking at the south pole must not sample row 0."""
    # Direct sampler: past-the-bottom y clamps to the bottom row.
    H, W = 10, 20
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[0, :, :] = 0.0
    img[-1, :, :] = 1000.0
    img[1:-1, :, :] = 500.0
    v = bilinear_sampling(img, float(H), 5.0, True)
    assert float(v[0]) > 900.0, f"bottom clamp expected ~1000, got {v}"

    # End-to-end: top row 0, bottom row 1000; pitch=+90 looks at south pole.
    equi = np.zeros((100, 200, 3), dtype=np.float32)
    equi[0, :, :] = 0.0
    equi[-1, :, :] = 1000.0
    equi[1:-1, :, :] = 500.0
    out = equi2pers(equi, output_width=32, output_height=32, fov_x=90,
                    yaw=0, pitch=90, roll=0,
                    use_gpu=False, output_dtype="float32")
    centre = float(out[16, 16, 0])
    assert centre > 900.0, f"south pole should sample bottom row ~1000, got {centre}"


def test_process_chunk_threads_output_dtype():
    """process_chunk accepts the 8-tuple with output_dtype (and legacy 7-tuple)."""
    equi = np.full((16, 32, 3), 100.0, dtype=np.float32)
    params = (90.0, 0.0, 0.0, 0.0)
    chunk_f, xs, xe = process_chunk(
        (equi, 0, 8, 16, 8, params, "bilinear", "float32"))
    assert chunk_f.dtype == np.float32
    assert np.allclose(chunk_f, 100.0, atol=1e-3)
    chunk_u, _, _ = process_chunk(
        (equi, 0, 8, 16, 8, params, "bilinear", "uint8"))
    assert chunk_u.dtype == np.uint8
    # Legacy 7-tuple still works (defaults to uint8).
    chunk_legacy, _, _ = process_chunk(
        (equi, 0, 8, 16, 8, params, "bilinear"))
    assert chunk_legacy.dtype == np.uint8
