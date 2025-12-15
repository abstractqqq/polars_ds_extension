import polars as pl
import polars_ds as pds
import pytest
from polars.testing import assert_frame_equal

def test_kaplan_meier():

    from sksurv.datasets import load_veterans_lung_cancer
    from sksurv.nonparametric import kaplan_meier_estimator    

    _, y = load_veterans_lung_cancer()
    time, prob_surv, conf_int = kaplan_meier_estimator(
        y["Status"], y["Survival_in_days"], conf_type="log-log"
    )

    df_result = pl.from_dict({
        "t": time
        , "prob": prob_surv
    })

    df = pl.from_dict({
        "status": y['Status']
        , "survival_time": y["Survival_in_days"]
    })

    df_pds_result = df.select(
        pds.query_kaplan_meier_prob(
            "status"
            , "survival_time"
        ).alias("estimate")
    ).unnest("estimate")

    assert_frame_equal(df_result, df_pds_result)

