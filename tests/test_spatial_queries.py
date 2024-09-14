import polars_ds as pds
import numpy as np
from polars_ds.spatial import KDTree as KDT
from scipy.spatial import KDTree


def test_kdtree():
    size = 2000
    df = pds.frame(size=size).with_columns(*(pds.random().alias(f"var{i}") for i in range(3)))
    X = df.select(f"var{i}" for i in range(3)).to_numpy(order="c")

    pds_tree = KDT(X, distance="l2")
    scipy_tree = KDTree(X, copy_data=True)

    distances_pds, indices_pds = pds_tree.knn(X, k=10, parallel=False)
    distances_scipy, indices_scipy = scipy_tree.query(X, k=10, p=2)

    assert np.all(distances_pds == distances_scipy)
    assert np.all(indices_pds.astype(np.int64) == indices_scipy.astype(np.int64))

    within_pds = pds_tree.within(X, r=0.1, sort=False)
    within_scipy = scipy_tree.query_ball_point(X, r=0.1, p=2, return_sorted=False)
    assert all(set(x) == set(y) for x, y in zip(within_pds, within_scipy))

    within_count = pds_tree.within_count(X, r=0.1)
    assert all(int(n) == len(pts) for n, pts in zip(within_count, within_pds))
