"""Remove concentric interference rings (e.g. laser fringes) from an image
while preserving non-radially-symmetric scene detail.

Pipeline:
  1. Estimate the ring center from the image gradient field. For a perfectly
     concentric pattern the gradient at every ring pixel points along the
     line from the center to that pixel, so the center minimizes
        sum_p w_p * ( g_y(p)*c_x - g_x(p)*c_y - (g_y*p_x - g_x*p_y) )^2
     which is a 2x2 linear system in c.
  2. Bin pixels by integer radius around the estimated center and take the
     median value per bin -> radial ring profile I_ring(r).
  3. Subtract I_ring(r) from each pixel and add back the global mean.
  4. (Optional) Warp to polar, apply a 1-D tangential Gaussian to suppress
     the residual high-frequency moire that the radial subtraction does not
     touch, then warp back.

Usage:
    python derings.py input.png output.png [--tangential-sigma 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _radial_component_energy(img_f: np.ndarray, cx: float, cy: float,
                             ys: np.ndarray, xs: np.ndarray) -> float:
    """Variance of the azimuthal mean per radius - purely-radial component energy.

    Pixels that form a coherent set of concentric rings about (cx, cy) all
    contribute to one radial bin, so this peaks at the true ring center.
    """
    r = np.round(np.hypot(xs - cx, ys - cy)).astype(np.int32)
    flat_r = r.ravel()
    flat_v = img_f.ravel()
    n_bins = int(flat_r.max()) + 1
    counts = np.bincount(flat_r, minlength=n_bins)
    sums = np.bincount(flat_r, weights=flat_v, minlength=n_bins)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    total = max(np.sum(counts), 1)
    weighted_mean = np.sum(sums) / total
    diffs = means - weighted_mean
    return float(np.sum(counts * diffs * diffs) / total)


def refine_center(gray: np.ndarray, cx0: float, cy0: float,
                  search_radius: int = 150, step: int = 6) -> tuple[float, float]:
    """Brute-force refine the center by maximizing radial-component energy.

    Operates on a bandpassed copy so only ring-scale structure is scored.
    """
    g = gray.astype(np.float64)
    low = cv2.GaussianBlur(g, (0, 0), sigmaX=2.5, sigmaY=2.5)
    very_low = cv2.GaussianBlur(g, (0, 0), sigmaX=40.0, sigmaY=40.0)
    bandpass = (low - very_low).astype(np.float64)

    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)

    def best_in(cx_range, cy_range):
        best = (-np.inf, (cx0, cy0))
        for cy in cy_range:
            for cx in cx_range:
                s = _radial_component_energy(bandpass, cx, cy, ys, xs)
                if s > best[0]:
                    best = (s, (cx, cy))
        return best

    # Coarse pass around initial estimate.
    cx_lo = int(round(cx0 - search_radius))
    cx_hi = int(round(cx0 + search_radius)) + 1
    cy_lo = int(round(cy0 - search_radius))
    cy_hi = int(round(cy0 + search_radius)) + 1
    _, (cx, cy) = best_in(range(cx_lo, cx_hi, step), range(cy_lo, cy_hi, step))

    # Finer pass.
    _, (cx, cy) = best_in(range(int(cx) - step, int(cx) + step + 1),
                          range(int(cy) - step, int(cy) + step + 1))
    # Half-pixel pass.
    _, (cx, cy) = best_in(np.arange(cx - 1.5, cx + 1.5, 0.5),
                          np.arange(cy - 1.5, cy + 1.5, 0.5))
    return float(cx), float(cy)


def estimate_center(
    gray: np.ndarray,
    mask: np.ndarray | None = None,
    low_sigma: float = 2.5,
    high_sigma: float = 40.0,
) -> tuple[float, float]:
    """Least-squares center from gradient orientation.

    Pre-filters with a band-pass tuned to ring-scale features so that finer
    moire and the slow background gradient do not bias the fit.
    """
    g = gray.astype(np.float64)
    low = cv2.GaussianBlur(g, (0, 0), sigmaX=low_sigma, sigmaY=low_sigma)
    very_low = cv2.GaussianBlur(g, (0, 0), sigmaX=high_sigma, sigmaY=high_sigma)
    bandpass = low - very_low

    gx = cv2.Sobel(bandpass, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(bandpass, cv2.CV_64F, 0, 1, ksize=5)

    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)

    g_mag2 = gx * gx + gy * gy
    g_mag = np.sqrt(g_mag2)

    # Unit gradient direction. Then the per-pixel residual
    #   uy*cx - ux*cy - (uy*px - ux*py)
    # is the perpendicular distance from c to the gradient line through p,
    # which is what we actually want to minimize (Hough-style intersection).
    safe = g_mag > 1e-6
    ux = np.where(safe, gx / np.where(safe, g_mag, 1.0), 0.0)
    uy = np.where(safe, gy / np.where(safe, g_mag, 1.0), 0.0)

    # Weight by gradient magnitude so ring pixels (large bandpass amplitude)
    # dominate over flat noise. Also drop the bottom ~85% of pixels entirely.
    weight = g_mag.copy()
    if mask is not None:
        weight = weight * mask
    if np.any(weight > 0):
        thresh = np.quantile(weight[weight > 0], 0.85)
        weight = np.where(weight >= thresh, weight, 0.0)

    def solve(w: np.ndarray) -> tuple[float, float]:
        a00 = np.sum(w * uy * uy)
        a01 = -np.sum(w * ux * uy)
        a11 = np.sum(w * ux * ux)
        rhs0 = np.sum(w * uy * (uy * xs - ux * ys))
        rhs1 = -np.sum(w * ux * (uy * xs - ux * ys))
        A = np.array([[a00, a01], [a01, a11]])
        b = np.array([rhs0, rhs1])
        return tuple(np.linalg.solve(A, b))

    cx, cy = solve(weight)

    # IRLS: re-weight by how well each pixel's gradient line passes through the
    # current center, then re-solve. Outlier scene gradients (text, scratches,
    # diagonal moire that survived the bandpass) get downweighted.
    for _ in range(4):
        residual = uy * (cx - xs) - ux * (cy - ys)
        # Robust scale estimate.
        med_abs_res = np.median(np.abs(residual[weight > 0]))
        if med_abs_res < 1e-9:
            break
        sigma = 1.4826 * med_abs_res
        # Tukey biweight: zero weight beyond 4*sigma, smooth falloff inside.
        u = residual / (4.0 * sigma)
        bisq = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        new_weight = weight * bisq
        if np.sum(new_weight) < 1e-9:
            break
        new_cx, new_cy = solve(new_weight)
        if abs(new_cx - cx) < 0.05 and abs(new_cy - cy) < 0.05:
            cx, cy = new_cx, new_cy
            break
        cx, cy = new_cx, new_cy

    return float(cx), float(cy)


def radial_profile_subtract(img: np.ndarray, center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the median radial profile around `center`. Returns (filtered, profile)."""
    h, w = img.shape
    cx, cy = center
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r_int = np.round(r).astype(np.int32)
    n_bins = int(r_int.max()) + 1

    flat_r = r_int.ravel()
    flat_v = img.ravel().astype(np.float64)
    order = np.argsort(flat_r, kind="stable")
    sorted_r = flat_r[order]
    sorted_v = flat_v[order]

    # Find bin boundaries in the sorted-by-radius array.
    edges = np.searchsorted(sorted_r, np.arange(n_bins + 1))

    profile = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        if b > a:
            profile[i] = np.median(sorted_v[a:b])

    ring_image = profile[r_int]
    out = img.astype(np.float64) - ring_image + img.mean()
    return out, profile


def tangential_smooth(img: np.ndarray, center: tuple[float, float], sigma: float) -> np.ndarray:
    """Smooth along constant-radius arcs to suppress azimuthal high-freq moire."""
    if sigma <= 0:
        return img
    h, w = img.shape
    cx, cy = center
    # Use the maximum corner distance so every image pixel maps inside.
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    max_r = float(max(np.hypot(px - cx, py - cy) for px, py in corners))
    src = np.ascontiguousarray(img, dtype=np.float32)
    polar = cv2.warpPolar(
        src,
        (int(max_r), 1440),  # (radius bins, angle bins)
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR,
    )
    # polar shape: (angle, radius). Smooth along the angle (vertical) axis only.
    # OpenCV won't accept BORDER_WRAP on the column border, so wrap manually.
    ksize = max(3, int(2 * round(3 * sigma) + 1) | 1)
    pad = ksize
    padded = np.vstack([polar[-pad:], polar, polar[:pad]])
    blurred_padded = cv2.GaussianBlur(padded, (1, ksize), sigmaX=0, sigmaY=sigma)
    blurred = blurred_padded[pad:pad + polar.shape[0]]
    back = cv2.warpPolar(
        blurred,
        (w, h),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR,
    )
    # Pixels that fell outside the polar grid get 0 from warp; fall back to original.
    invalid = ~np.isfinite(back) | (back == 0)
    back = np.where(invalid, src, back)
    return back


def derings(
    img: np.ndarray,
    tangential_sigma: float = 0.0,
    overlay_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Main entry point. Returns (filtered_uint8, info_dict)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray_f = gray.astype(np.float64)

    # If the caller flagged on-screen overlays (HUD text), exclude them from the fit.
    fit_mask = None
    if overlay_mask is not None:
        fit_mask = (overlay_mask == 0).astype(np.float64)

    cx, cy = estimate_center(gray_f, fit_mask)
    # Gradient-line fitting can be pulled off-axis by aliased moire or scene
    # gradients that aren't actually radial. Refine by directly maximizing the
    # radial-component energy in a local neighborhood.
    cx, cy = refine_center(gray_f, cx, cy)

    filtered, profile = radial_profile_subtract(gray_f, (cx, cy))
    if tangential_sigma > 0:
        filtered = tangential_smooth(filtered, (cx, cy), tangential_sigma)

    # Restore overlay pixels untouched if a mask was provided.
    if overlay_mask is not None:
        filtered = np.where(overlay_mask > 0, gray_f, filtered)

    out = np.clip(filtered, 0, 255).astype(np.uint8)
    return out, {"center": (cx, cy), "profile": profile}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--tangential-sigma", type=float, default=0.0,
                   help="Gaussian sigma (in angular pixels) for optional moire suppression.")
    args = p.parse_args()

    img = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not read {args.input}")
    out, info = derings(img, tangential_sigma=args.tangential_sigma)
    cv2.imwrite(str(args.output), out)
    cx, cy = info["center"]
    print(f"Estimated ring center: ({cx:.1f}, {cy:.1f})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
