# MoE Dukascopy Pipeline

This project downloads 5 years of Dukascopy minute-bar data for:

- `US100` via `usatechidxusd`
- `US500` via `usa500idxusd`

The pipeline then:

- enriches each row with UTC calendar and session fields
- removes malformed, duplicate, and zero-volume rows
- preserves true market gaps as explicit features
- emits cleaned CSVs and analysis reports for model training

Run:

```powershell
npm run pipeline
```

Outputs:

- raw CSV files in `data/raw/`
- cleaned training CSV files in `data/processed/`
- summary JSON and Markdown reports in `reports/`
