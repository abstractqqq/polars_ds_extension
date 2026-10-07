import polars as pl
import polars_ds as pds


def test_lin_reg_multichunk_weights():
    parts = [
        pl.DataFrame({"x": [1.0, 2.0], "y": [2.0, 4.0], "w": [1.0, 1.0]}),
        pl.DataFrame({"x": [3.0, 4.0], "y": [6.0, 8.0], "w": [1.0, 1.0]}),
    ]
    df = pl.concat(parts, rechunk=False)
    assert df["w"].n_chunks() == 2

    # lin_reg
    coeffs = df.select(pds.lin_reg("x", target="y", weights="w"))[0, 0]
    assert len(coeffs) > 0

    # lin_reg_pred
    pred = df.select(pds.lin_reg("x", target="y", weights="w", return_pred=True))
    assert len(pred) == 4

    # lin_reg_report
    rep = df.select(pds.lin_reg_report("x", target="y", weights="w"))
    assert len(rep) > 0


def test_knn_multichunk_inputs():
    parts = [
        pl.DataFrame({"id": [0, 1], "x": [1.0, 2.0], "y": [1.0, 2.0]}),
        pl.DataFrame({"id": [2, 3], "x": [10.0, 20.0], "y": [10.0, 20.0]}),
    ]
    df = pl.concat(parts, rechunk=False)
    assert df["id"].n_chunks() == 2

    # query_knn_ptwise
    res_ptwise = df.select(pds.query_knn_ptwise("x", "y", index="id", k=2))
    assert len(res_ptwise) == 4

    # query_knn_ptwise_w_dist
    res_w_dist = df.select(pds.query_knn_ptwise("x", "y", index="id", k=2, return_dist=True))
    assert len(res_w_dist) == 4

    # query_knn_avg (target is id)
    res_avg = df.select(pds.query_knn_avg("x", "y", target="id", k=2))
    assert len(res_avg) == 4

    # query_radius_ptwise
    res_rad = df.select(pds.query_radius_ptwise("x", "y", index="id", r=3.0))
    assert len(res_rad) == 4

    # query_radius_ptwise_null_safe
    res_rad_ns = df.select(pds.query_radius_ptwise_null_safe("x", "y", index="id", r=3.0))
    assert len(res_rad_ns) == 4


def test_knn_index_with_nulls_errors():
    import pytest

    df_null_id = pl.DataFrame({"id": [0, None, 2], "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(Exception, match="cannot contain null"):
        df_null_id.select(pds.query_knn_ptwise("x", "y", index="id", k=2))


def test_psi_multichunk():
    parts = [
        pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.5, 2.5, 3.5]}),
        pl.DataFrame({"a": [4.0, 5.0, 6.0], "b": [4.5, 5.5, 6.5]}),
    ]
    df = pl.concat(parts, rechunk=False)
    assert df["a"].n_chunks() == 2
    assert df["b"].n_chunks() == 2
    res = df.select(pds.psi_w_breakpoints("a", "b", breakpoints=[2.0, 4.0]))
    assert len(res) > 0


def test_trapz_multichunk():
    parts = [
        pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}),
        pl.DataFrame({"x": [2.0, 3.0], "y": [2.0, 3.0]}),
    ]
    df = pl.concat(parts, rechunk=False)
    assert df["y"].n_chunks() == 2
    # scalar dx
    res1 = df.select(pds.integrate_trapz("y", x=1.0))
    # expr x
    res2 = df.select(pds.integrate_trapz("y", x="x"))
    assert res1[0, 0] > 0
    assert res2[0, 0] > 0


def test_convolve_multichunk():
    parts = [
        pl.DataFrame({"x": [1.0, 2.0, 3.0]}),
        pl.DataFrame({"x": [4.0, 5.0, 6.0]}),
    ]
    df = pl.concat(parts, rechunk=False)
    assert df["x"].n_chunks() == 2

    k_parts = [
        pl.DataFrame({"k": [0.5]}),
        pl.DataFrame({"k": [0.5]}),
    ]
    df_k = pl.concat(k_parts, rechunk=False)
    assert df_k["k"].n_chunks() == 2

    res1 = df.select(pds.convolve("x", kernel=df_k["k"]))
    assert len(res1) > 0

    res2 = df.select(pds.convolve("x", kernel=[0.5, 0.5]))
    assert len(res2) > 0


def test_add_at_multichunk():
    parts = [
        pl.DataFrame({"idx": [0, 1], "val": [10.0, 20.0]}),
        pl.DataFrame({"idx": [1, 2], "val": [30.0, 40.0]}),
    ]
    df = pl.concat(parts, rechunk=False)
    assert df["idx"].n_chunks() == 2
    assert df["val"].n_chunks() == 2

    res = df.select(pds.add_at("idx", "val", buffer_size=3))
    assert len(res) == 3


def test_smooth_spline_multichunk():
    parts = [
        pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 5.0]}),
        pl.DataFrame({"x": [4.0, 5.0, 6.0], "y": [7.0, 11.0, 13.0]}),
    ]
    df = pl.concat(parts, rechunk=False)
    assert df["x"].n_chunks() == 2
    assert df["y"].n_chunks() == 2

    res = df.select(pds.smooth_spline("x", "y", lambda_=0.1))
    assert len(res) == 6


def test_subseq_sim_cnt_multichunk():
    parts = [
        pl.DataFrame({"t": [1.0, 2.0, 3.0, 4.0]}),
        pl.DataFrame({"t": [2.0, 3.0, 4.0, 5.0]}),
    ]
    df = pl.concat(parts, rechunk=False)
    assert df["t"].n_chunks() == 2

    query = [2.0, 3.0]
    res_l2 = df.select(pds.query_similar_count(query=query, target="t", metric="sql2", threshold=0.1))
    assert len(res_l2) == 1

    res_zl2 = df.select(pds.query_similar_count(query=query, target="t", metric="sqzl2", threshold=0.1))
    assert len(res_zl2) == 1
