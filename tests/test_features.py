import pandas as pd

from src.features.build_features import (
    FEATURE_SETS,
    add_features,
    add_history_features,
    time_split,
)

POST_TRANSACTION_COLS = ["newbalanceOrig", "newbalanceDest", "errorOrig", "errorDest", "emptiedOrig"]


def make_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 1, 2, 30],
            "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT"],
            "amount": [100.0, 50.0, 500.0, 20.0],
            "nameOrig": ["A", "A", "A", "B"],
            "oldbalanceOrg": [100.0, 300.0, 500.0, 1000.0],
            "newbalanceOrig": [0.0, 250.0, 0.0, 980.0],
            "nameDest": ["X", "Y", "X", "Z"],
            "oldbalanceDest": [0.0, 10.0, 100.0, 0.0],
            "newbalanceDest": [0.0, 60.0, 600.0, 20.0],
            "isFraud": [1, 0, 1, 0],
        }
    )


def test_features_for_a_transfer_that_drains_the_sender():
    row = add_features(make_transactions()).iloc[0]
    assert row["would_empty_orig"] == 1
    assert row["dest_was_empty"] == 1
    assert row["is_transfer"] == 1
    assert row["hour_of_day"] == 1
    assert row["errorOrig"] == 0
    assert abs(row["amount_to_balance"] - 100 / 101) < 1e-9


def test_features_for_a_normal_cash_out():
    row = add_features(make_transactions()).iloc[1]
    assert row["would_empty_orig"] == 0
    assert row["dest_was_empty"] == 0
    assert row["is_transfer"] == 0


def test_serving_features_never_use_post_transaction_columns():
    """Leakage guard: the live API cannot know balances after the transaction."""
    for name in ["serving", "history"]:
        leaked = set(FEATURE_SETS[name]) & set(POST_TRANSACTION_COLS)
        assert not leaked, f"{name} feature set leaks future information: {leaked}"


def test_time_split_is_ordered_and_does_not_overlap():
    df = pd.DataFrame({"step": [1, 349, 350, 399, 400, 700]})
    train, valid, test = time_split(df)
    assert train["step"].tolist() == [1, 349]
    assert valid["step"].tolist() == [350, 399]
    assert test["step"].tolist() == [400, 700]


def test_history_counts_only_earlier_steps():
    full = make_transactions()
    result = add_history_features(add_features(full), full)
    # Account A sends twice in step 1, then once in step 2.
    assert result["orig_prev_count"].tolist() == [0, 0, 2, 0]
    # Account X receives at step 1 and again at step 2; the step-2 row sees only step 1.
    assert result["dest_prev_count"].tolist() == [0, 0, 1, 0]
    assert result["dest_prev_amount"].tolist() == [0.0, 0.0, 100.0, 0.0]
