#!/usr/bin/env python3
"""
Convert a real fUS recording (.mat or HDF5) into the npkit .npz format.

Run this ON A MACHINE THAT HAS scipy / h5py (e.g. your laptop in the project
venv) — the decoding sandbox is NumPy-only by design. The output .npz is then
read by run_dataset.py with no further conversion.

The .npz must expose at least:
    X : (n_trials, n_voxels, n_timepoints)   float
    y : (n_trials,)                          integer/string class labels

and optionally TR, baseline length, class names, and 2-D map shape (stored as
plain numeric/string arrays — never pickled objects).

Examples
--------
  # MATLAB v7 .mat with variables 'data' (T,V,trials) and 'labels'
  python scripts/convert_to_npz.py in.mat out.npz \
      --x-var data --y-var labels --axes t v n --tr 0.4 --baseline 4

  # HDF5
  python scripts/convert_to_npz.py in.h5 out.npz --x-var /fus/X --y-var /fus/y
"""
import argparse
import sys
import numpy as np


def _load_mat(path, x_var, y_var):
    try:
        from scipy.io import loadmat
        m = loadmat(path)
        return np.asarray(m[x_var]), np.asarray(m[y_var]).squeeze()
    except NotImplementedError:
        # v7.3 .mat is actually HDF5
        return _load_h5(path, x_var, y_var)


def _load_h5(path, x_var, y_var):
    import h5py
    with h5py.File(path, "r") as f:
        return np.asarray(f[x_var]), np.asarray(f[y_var]).squeeze()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--x-var", required=True, help="variable/dataset name for X")
    ap.add_argument("--y-var", required=True, help="variable/dataset name for y")
    ap.add_argument("--axes", nargs=3, default=["n", "v", "t"],
                    help="order of X axes in the source; target is n v t "
                         "(trials voxels timepoints). e.g. --axes t v n")
    ap.add_argument("--tr", type=float, default=1.0)
    ap.add_argument("--baseline", type=int, default=3)
    ap.add_argument("--class-names", nargs="*", default=None)
    ap.add_argument("--spatial-shape", nargs=2, type=int, default=None,
                    help="rows cols, if voxels form a 2-D image (for weight maps)")
    args = ap.parse_args()

    if args.input.endswith(".mat"):
        X, y = _load_mat(args.input, args.x_var, args.y_var)
    elif args.input.endswith((".h5", ".hdf5")):
        X, y = _load_h5(args.input, args.x_var, args.y_var)
    else:
        sys.exit("input must be .mat, .h5, or .hdf5")

    # reorder axes to (trials, voxels, timepoints)
    src = {ax: i for i, ax in enumerate(args.axes)}
    X = np.transpose(X, (src["n"], src["v"], src["t"]))
    print(f"X -> {X.shape} (trials, voxels, timepoints); y -> {y.shape}")

    # remap labels to contiguous ints; keep a name table if labels were strings
    classes, y_int = np.unique(y, return_inverse=True)
    names = args.class_names or [str(c) for c in classes]

    fields = {
        "X": X.astype(np.float32),
        "y": y_int.astype(np.int64),
        "tr": np.float64(args.tr),
        "baseline_timepoints": np.int64(args.baseline),
        "class_names": np.array(names, dtype=np.str_),
        "source": np.str_(args.input),
    }
    if args.spatial_shape:
        fields["spatial_shape"] = np.array(args.spatial_shape, dtype=np.int64)

    np.savez_compressed(args.output, **fields)
    print(f"wrote {args.output}  ({len(classes)} classes: {names})")


if __name__ == "__main__":
    main()
