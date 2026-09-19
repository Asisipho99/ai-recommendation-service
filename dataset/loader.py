"""Load the raw external dataset into a canonical frame.

The two published releases of the retail dataset use different column names:

    Online Retail II (2009-2011)  Invoice, StockCode, Description, Quantity,
                                  InvoiceDate, Price, Customer ID, Country
    Online Retail   (2010-2011)  InvoiceNo, StockCode, Description, Quantity,
                                  InvoiceDate, UnitPrice, CustomerID, Country

Both are normalised to snake_case here so nothing downstream has to care which
file the user happened to download, and a Kaggle copy works as well as the UCI
one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# Every spelling seen across the releases and the Kaggle mirrors.
COLUMN_ALIASES = {
    "invoice": "invoice",
    "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "unit_price",
    "unitprice": "unit_price",
    "customerid": "customer_id",
    "customer id": "customer_id",
    "country": "country",
}

REQUIRED = ["invoice", "stock_code", "quantity", "invoice_date", "unit_price"]


class DatasetNotFound(FileNotFoundError):
    """Raised when no raw dataset file is present."""


def candidate_files() -> list[Path]:
    """Raw dataset files, most recently modified first."""
    if not RAW_DIR.exists():
        return []
    files = [
        path
        for path in RAW_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".gz", ".csv", ".xlsx", ".xls"}
        and not path.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        renamed[column] = COLUMN_ALIASES.get(key, key.replace(" ", "_"))
    frame = frame.rename(columns=renamed)

    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(
            f"dataset is missing expected columns {missing}; "
            f"found {list(frame.columns)}"
        )
    return frame


def load_raw(limit: int | None = None) -> pd.DataFrame:
    """Read the raw dataset.

    `limit` caps the number of rows, which keeps memory sane on a small
    container. The full file is around a million rows.
    """
    files = candidate_files()
    if not files:
        raise DatasetNotFound(
            f"No dataset in {RAW_DIR}. Run: python scripts/download_dataset.py"
        )

    path = files[0]
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, nrows=limit, engine="openpyxl")
    else:
        compression = "gzip" if path.suffix.lower() == ".gz" else None
        frame = pd.read_csv(
            path,
            nrows=limit,
            compression=compression,
            encoding="latin-1",
            low_memory=False,
        )

    frame = normalise_columns(frame)
    frame.attrs["source_file"] = path.name
    return frame


def source_info() -> dict:
    """Describe what is on disk, without parsing it."""
    files = candidate_files()
    if not files:
        return {"available": False, "path": str(RAW_DIR), "files": []}

    return {
        "available": True,
        "path": str(RAW_DIR),
        "files": [
            {
                "name": f.name,
                "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
            }
            for f in files
        ],
        "active": files[0].name,
    }
