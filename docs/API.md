# API and data documentation

## Scoring service

Base URL: `http://localhost:8000` (use `kubectl port-forward svc/scoring 8000:8000` for the Kubernetes deployment). Interactive docs: `/docs`.

### `GET /health`

```json
{"status": "ok", "feature_set": "history", "threshold": 0.8791942596435547}
```

### `POST /score`

Request body (all fields required):

| Field | Type | Meaning |
|---|---|---|
| `step` | int | Hour of the simulation (1-743) |
| `type` | string | `TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN` or `DEBIT` |
| `amount` | float | Transaction amount |
| `nameOrig` | string | Sender account id |
| `oldbalanceOrg` | float | Sender balance before the transaction |
| `nameDest` | string | Receiver account id |
| `oldbalanceDest` | float | Receiver balance before the transaction |

Only `TRANSFER` and `CASH_OUT` are scored (fraud appears only in these types). Other types are remembered for account history and return `{"scored": false}`.

Example response for a scored transaction:

```json
{
  "scored": true,
  "fraud_probability": 0.908646,
  "is_fraud": true,
  "threshold": 0.8791942596435547,
  "latency_ms": 4.1
}
```

An incomplete body returns HTTP 422.

Example call (PowerShell):

```powershell
$body = @{ step=1; type="TRANSFER"; amount=181.0; nameOrig="C1305486145"; oldbalanceOrg=181.0; nameDest="C553264065"; oldbalanceDest=0.0 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/score -Method Post -Body $body -ContentType "application/json"
```

## Model features (`history` feature set)

| Feature | Definition |
|---|---|
| `amount` | Transaction amount |
| `oldbalanceOrg`, `oldbalanceDest` | Balances before the transaction |
| `amount_to_balance` | `amount / (oldbalanceOrg + 1)` |
| `would_empty_orig` | 1 if the amount is at least the sender's balance |
| `dest_was_empty` | 1 if the receiver's balance was 0 |
| `hour_of_day` | `step % 24` |
| `is_transfer` | 1 for TRANSFER, 0 for CASH_OUT |
| `orig_prev_count` | Transactions the sender made in earlier steps |
| `dest_prev_count`, `dest_prev_amount` | Count and total amount the receiver got in earlier steps |

## Feast feature view `account_stats`

Entity `account_id`. Fields: `sent_count` (int), `received_count` (int), `received_amount` (float). Values are cumulative and stamped one hour after each step, so a lookup at step t returns only activity before t.

## Kafka message (topic `transactions`)

The seven request fields above, plus `isFraud` (the true label, kept for evaluation only and never sent to the model) and `sent_at` (producer timestamp, used for end-to-end latency).

## SQLite tables

`data/predictions.db`, table `predictions`: `step, type, amount, nameOrig, oldbalanceOrg, nameDest, oldbalanceDest, fraud_probability, predicted_fraud, true_label, service_latency_ms, end_to_end_ms, scored_at`.

`data/cases.db`, table `cases`: `case_id, status (new | investigating | closed), analyst_label (unreviewed | confirmed fraud | false alarm), note, updated_at`.
