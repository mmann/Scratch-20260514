"""Synthetic test for derings.py.

Builds a 1080x1080 grayscale image that mimics the user's laser-fringe
photo: a slowly-varying background, concentric rings with center near the
lower-left of the frame, a fine high-frequency moire pattern, and some
scene detail (small dots, thin lines, text-like blobs) that we don't want
the filter to remove. Then it runs derings() and prints metrics.

Metrics:
  - Recovered center vs. ground truth.
  - Ring-energy ratio: variance of an annular high-pass band, before vs.
    after. Lower is better.
  - Detail preservation: PSNR between the recovered "scene only" image
    and the ground-truth scene-only image, computed on detail pixels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from derings import derings, estimate_center, radial_profile_subtract


def build_synthetic(h: int = 1080, w: int = 1080, seed: int = 0) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Return (composite_uint8, scene_only_uint8, (cx, cy))."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)

    # 1. Smooth background.
    background = 110 + 20 * np.cos(2 * np.pi * (xs / w) * 0.7 + 0.3) \
                     + 15 * np.sin(2 * np.pi * (ys / h) * 0.5 - 0.7)

    # 2. Concentric rings centered near the lower-left, like in the photo.
    cx, cy = 120.0, 760.0
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    # Period grows slowly with r (chirp-like) and amplitude tapers a touch.
    rings = 22 * np.cos(2 * np.pi * r / 10.0 - 0.0015 * r) * np.exp(-r / 2400.0)

    # 3. Fine diagonal moire (high-freq aliased fringe).
    moire = 6 * np.cos(2 * np.pi * (xs + ys) / 2.6)

    # 4. Scene detail we want to preserve. None of it is centered at (cx, cy).
    scene = np.zeros_like(background)

    # A handful of small bright/dark dots scattered over the frame.
    for _ in range(40):
        py = int(rng.integers(40, h - 40))
        px = int(rng.integers(40, w - 40))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        radius = int(rng.integers(2, 5))
        cv2.circle(scene, (px, py), radius, sign * 35.0, thickness=-1)

    # A few thin lines at varied orientations.
    for _ in range(6):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(scene, p1, p2, 25.0, thickness=1, lineType=cv2.LINE_AA)

    # Some "text-like" blobs in the upper-left corner, mimicking the HUD.
    for i, label in enumerate(["19.7 fps", "37.50 C"]):
        cv2.putText(scene, label, (10, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 90.0, 2, cv2.LINE_AA)

    # 5. A little sensor noise.
    noise = rng.normal(0.0, 1.5, size=background.shape)

    composite = background + rings + moire + scene + noise
    composite_u8 = np.clip(composite, 0, 255).astype(np.uint8)

    # The "scene only" reference is background + scene (no rings, no moire, no noise),
    # so we can score detail preservation against a known truth.
    reference = np.clip(background + scene, 0, 255).astype(np.uint8)
    return composite_u8, reference, (cx, cy)


def annular_highpass_energy(img: np.ndarray, center: tuple[float, float]) -> float:
    """Variance of the band-pass that contains the ring frequencies (period ~10 px)."""
    g = img.astype(np.float32)
    low = cv2.GaussianBlur(g, (0, 0), sigmaX=8.0, sigmaY=8.0)
    high = g - low
    return float(np.var(high))


def radial_component_energy(img: np.ndarray, center: tuple[float, float]) -> float:
    """Energy in the purely-radial component: variance of the azimuthal mean per radius.

    Anything that's a function of radius alone (i.e., the ring artifact) lives entirely
    here. Scene detail and moire average out across each ring and contribute almost
    nothing.
    """
    h, w = img.shape
    cx, cy = center
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.round(np.hypot(xs - cx, ys - cy)).astype(np.int32)
    n_bins = int(r.max()) + 1
    flat_r = r.ravel()
    flat_v = img.astype(np.float64).ravel()
    counts = np.bincount(flat_r, minlength=n_bins)
    sums = np.bincount(flat_r, weights=flat_v, minlength=n_bins)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    # Energy weighted by pixels per ring, demeaned.
    weighted_mean = np.sum(sums) / max(np.sum(counts), 1)
    diffs = means - weighted_mean
    return float(np.sum(counts * diffs * diffs) / max(np.sum(counts), 1))


def detail_psnr(reference: np.ndarray, candidate: np.ndarray, scene_mask: np.ndarray) -> float:
    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    diff = diff[scene_mask]
    mse = float(np.mean(diff * diff))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def main() -> None:
    out_dir = Path(__file__).parent / "test_out"
    out_dir.mkdir(exist_ok=True)

    composite, reference, true_center = build_synthetic()
    cv2.imwrite(str(out_dir / "input.png"), composite)
    cv2.imwrite(str(out_dir / "reference.png"), reference)

    # Center recovery.
    est_cx, est_cy = estimate_center(composite.astype(np.float64))
    ctr_err = float(np.hypot(est_cx - true_center[0], est_cy - true_center[1]))
    print(f"True center:        ({true_center[0]:.1f}, {true_center[1]:.1f})")
    print(f"Estimated center:   ({est_cx:.1f}, {est_cy:.1f})")
    print(f"Center error:       {ctr_err:.2f} px")

    # Run the filter (radial subtraction only).
    filtered_radial, info = derings(composite, tangential_sigma=0.0)
    cv2.imwrite(str(out_dir / "filtered_radial.png"), filtered_radial)

    # Run the filter with tangential smoothing for moire.
    filtered_full, _ = derings(composite, tangential_sigma=1.2)
    cv2.imwrite(str(out_dir / "filtered_full.png"), filtered_full)

    # Band-pass energy (rings + moire + noise in the ring-frequency band).
    e_in = annular_highpass_energy(composite, info["center"])
    e_radial = annular_highpass_energy(filtered_radial, info["center"])
    e_full = annular_highpass_energy(filtered_full, info["center"])
    print(f"Band-pass variance:      input={e_in:8.1f}  radial={e_radial:8.1f}  +tangential={e_full:8.1f}")
    print(f"  reduction radial-only:           {(1 - e_radial / e_in) * 100:5.1f}%")
    print(f"  reduction with tangential pass:  {(1 - e_full / e_in) * 100:5.1f}%")

    # Pure radial-component energy - this is what the radial subtraction is for.
    rc_in = radial_component_energy(composite, info["center"])
    rc_radial = radial_component_energy(filtered_radial, info["center"])
    rc_full = radial_component_energy(filtered_full, info["center"])
    print(f"Radial-component energy: input={rc_in:8.2f}  radial={rc_radial:8.2f}  +tangential={rc_full:8.2f}")
    print(f"  reduction radial-only:           {(1 - rc_radial / rc_in) * 100:5.1f}%")

    # Detail preservation: compare against scene-only reference, on a mask
    # that excludes pixels close to the ring center where errors don't matter.
    h, w = composite.shape
    ys, xs = np.mgrid[0:h, 0:w]
    far = np.hypot(xs - true_center[0], ys - true_center[1]) > 60
    psnr_radial = detail_psnr(reference, filtered_radial, far)
    psnr_full = detail_psnr(reference, filtered_full, far)
    psnr_input = detail_psnr(reference, composite, far)
    print(f"PSNR vs scene-only reference (higher = better):")
    print(f"  unfiltered input:        {psnr_input:5.2f} dB")
    print(f"  radial-only filter:      {psnr_radial:5.2f} dB")
    print(f"  + tangential moire pass: {psnr_full:5.2f} dB")

    # Sanity assertions so the test fails loudly on regressions.
    assert ctr_err < 3.0, f"center error too large: {ctr_err}"
    assert rc_radial < 0.05 * rc_in, \
        f"radial subtraction failed to remove ring component: {rc_radial:.2f} of {rc_in:.2f}"
    assert e_radial < 0.4 * e_in, \
        f"band-pass energy not reduced enough: {e_radial:.1f} of {e_in:.1f}"
    assert psnr_radial >= psnr_input, \
        f"filter degraded scene PSNR: input {psnr_input:.2f} -> filtered {psnr_radial:.2f}"
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
