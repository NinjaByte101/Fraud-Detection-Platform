from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

account = Entity(name="account", join_keys=["account_id"])

account_source = FileSource(
    path="../data/account_stats.parquet",
    timestamp_field="event_timestamp",
)

account_stats = FeatureView(
    name="account_stats",
    entities=[account],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sent_count", dtype=Int64),
        Field(name="received_count", dtype=Int64),
        Field(name="received_amount", dtype=Float64),
    ],
    source=account_source,
)