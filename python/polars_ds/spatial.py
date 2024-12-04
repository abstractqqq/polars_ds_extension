"""
Data structures for Spatial Queries, suah as KNN, within radius searches, etc.. These are good for small and medium sized data, 
and data of relatively small dimension (<30). 
"""

from __future__ import annotations

import numpy as np
from polars_ds._polars_ds import PyKDT
from typing import List, Tuple
from .typing import KdtDistance

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:  # 3.10, 3.9, 3.8
    from typing_extensions import Self


class KDTree:

    """
    A Kdtree. This copies data. It is not recommended to use Kdtree on data with dimension > 30. Query speed
    for higher dimensional data is not much better than brute force.
    """

    def __init__(self, X: np.ndarray, distance: KdtDistance = "sql2"):
        """
        Constructs a Kdtree by copying data from a NumPy matrix, where each row is a record with numbers
        in the row being the features.

        The rows will be indexed by [0..row_count(X)).

        Parameters
        ----------
        X
            The data matrix. The rows will be the points.
        distance
            One of 'l1', 'l2', 'sql2', 'inf'
        """
        if X.ndim != 2:
            raise ValueError("Input `X` is not a matrix.")
        if distance not in ("l1", "l2", "sql2", "inf"):
            raise ValueError(f"The given distance `{distance}` is not supported.")

        if X.flags["C_CONTIGUOUS"]:
            self.kdt = PyKDT(X, distance=distance)
        else:
            import warnings

            warnings.warn(
                "Input matrix is not c-contiguous, a copy is made internally and will take up more memory temporarily.",
                stacklevel=2,
            )
            self.kdt = PyKDT(np.ascontiguousarray(X), distance=distance)

    def __len__(self) -> int:
        return self.kdt.count()

    def count(self) -> int:
        return self.kdt.count()

    def add(self, X: np.ndarray) -> Self:
        """
        Appends the new data the tree. The new data will be index by [len(self)..len(self)+row_count(X)).

        Parameters
        ----------

        """
        if X.ndim != 2:
            raise ValueError("Input `X` is not a matrix.")

        self.kdt.add(X)
        return self

    def knn(
        self,
        X: np.ndarray,
        k: int,
        epsilon: float = 0.0,
        max_dist_bound: float = 9999.0,
        parallel: bool = False,
    ) -> (np.ndarray, np.ndarray):
        """
        Returns the K-nearest Neighbors and the distances in two matrices. The first
        is their indices and the second is the corresponding distances.

        Note that if max_dist_bound is low, it is possible that you get < k nearest neighbors.
        In that case, The output matrices will still be the right size but values will be padded. u32::MAX will
        be the index, which is not a valid index in almost all cases and the corresponding distance will
        be inf. You may detect this by ignoring indices > len(tree).

        Parameters
        ----------
        X
            The data matrix (points) to query the KNN for
        k
            Must be a positive int >= 1.
        epsilon
            If non-zero, then it is possible to miss a neighbor within epsilon distance.
        max_dist_bound
            Do not include neighbors beyond this distance.
        parallel
            Whether to run this in parallel or not
        """

        if X.ndim != 2:
            raise ValueError("Input `X` is not a matrix.")

        return self.kdt.knn(
            X, k, epsilon=abs(epsilon), max_dist_bound=abs(max_dist_bound), parallel=parallel
        )

    def within(
        self,
        X: np.ndarray,
        r: float | np.ndarray,
        sort: bool = False,
        return_dist: bool = False,
        parallel: bool = False,
    ) -> List[List[int]] | List[List[Tuple[int, float]]]:
        """
        Returns all points in the tree within radius r from points in X. For each point in X,
        this returns a list of indices of points in the tree that are within the distance from the point.
        If return_dist is true, for each point this will return a list of (int, float) tuple, which
        represents (index, distance).

        Parameters
        ----------
        X
            The data matrix (points) to query the KNN for
        r
            The radius. Either a scalar float, or a 1d array with len = row_count(X).
        sort
            Whether the result in each list should be sorted by distance. If true, this will take a bit longer to run.
        return_dist
            Whether to return only indices or (indices, distance) tuples for each row
        parallel
            Whether to evaluate this in parallel
        """
        if X.ndim != 2:
            raise ValueError("Input `X` is not a matrix.")

        if isinstance(r, (int, float)):
            rad = float(r)
            if return_dist:
                return self.kdt.within_with_dist(X, rad, sort, parallel)
            else:
                return self.kdt.within_idx_only(X, rad, sort, parallel)
        elif isinstance(r, np.ndarray):
            rad = np.ravel(r)
            if len(rad) == 1:
                self.within(X, rad[0], sort, parallel)
            else:
                if return_dist:
                    return self.kdt.within_with_dist_vec_r(X, rad, sort, parallel)
                else:
                    return self.kdt.within_idx_only_vec_r(X, rad, sort, parallel)
        else:
            raise ValueError("Input `r` has to be a scalar or a NumPy array.")

    def within_count(
        self, X: np.ndarray, r: float | np.ndarray, parallel: bool = False
    ) -> List[int]:
        """
        Returns the count of points in the tree within radius r from points in X. This is faster
        than calling within and then count the length of each output.

        Parameters
        ----------
        X
            The data matrix (points) to query the KNN for
        r
            The radius. Either a scalar float, or a 1d array with len = row_count(X).
        parallel
            Whether to evaluate this in parallel
        """
        if X.ndim != 2:
            raise ValueError("Input `X` is not a matrix.")

        if isinstance(r, (int, float)):
            return self.kdt.within_count(X, float(r), parallel)
        elif isinstance(r, np.ndarray):
            rad = np.ravel(r)
            if len(rad) == 1:
                self.within_count(X, rad[0], parallel)
            else:
                return self.kdt.within_count_vec_r(X, rad, parallel)
        else:
            raise ValueError("Input `r` has to be a scalar or a NumPy array.")
