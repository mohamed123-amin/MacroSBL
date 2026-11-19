import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BOOK_COLUMNS = ["title", "author", "isbn"]
MEMBER_COLUMNS = ["name", "member_id", "active"]
LOAN_COLUMNS = ["member_id", "isbn"]
TRAINING_COLUMNS = ["title", "author", "isbn", "is_available", "loan_count"]


def _get_records(data, key):
    records = data.get(key, [])
    if not isinstance(records, list):
        raise ValueError(f"'{key}' must be a list.")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Every item in '{key}' must be an object.")
    return records


def _text_column(frame, name):
    if name not in frame:
        frame[name] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame[name] = frame[name].astype("string").str.strip()
    frame[name] = frame[name].replace("", pd.NA)


def _clean_books(records):
    books = pd.json_normalize(records)
    for column in BOOK_COLUMNS:
        _text_column(books, column)

    missing_required = int(books[BOOK_COLUMNS].isna().any(axis=1).sum())
    books = books.dropna(subset=BOOK_COLUMNS).copy()
    duplicate_isbns = int(books.duplicated(subset=["isbn"]).sum())
    books = books.drop_duplicates(subset=["isbn"], keep="first")
    return books[BOOK_COLUMNS].reset_index(drop=True), {
        "books_missing_required_fields": missing_required,
        "duplicate_book_isbns": duplicate_isbns,
    }


def _clean_members(records):
    members = pd.json_normalize(records)
    _text_column(members, "name")
    if "member_id" not in members:
        members["member_id"] = pd.NA
    member_ids = pd.to_numeric(members["member_id"], errors="coerce")
    valid_ids = member_ids.notna() & (member_ids > 0) & (member_ids % 1 == 0)
    valid_rows = members["name"].notna() & valid_ids
    missing_or_invalid = int((~valid_rows).sum())
    members = members.loc[valid_rows].copy()
    members["member_id"] = member_ids.loc[valid_rows].astype("int64")

    if "active" not in members:
        members["active"] = pd.Series(True, index=members.index)
    active_values = members["active"]

    def parse_active(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        if pd.isna(value):
            return True
        return pd.NA

    active_statuses = active_values.map(parse_active)
    invalid_active_statuses = int(active_statuses.isna().sum())
    valid_active = active_statuses.notna()
    members = members.loc[valid_active].copy()
    members["active"] = active_statuses.loc[valid_active].astype(bool)

    duplicate_ids = int(members.duplicated(subset=["member_id"]).sum())
    members = members.drop_duplicates(subset=["member_id"], keep="first")
    cleaned = members[MEMBER_COLUMNS].reset_index(drop=True)
    return cleaned, {
        "members_missing_or_invalid_required_fields": missing_or_invalid,
        "members_with_invalid_active_status": invalid_active_statuses,
        "duplicate_member_ids": duplicate_ids,
    }


def _clean_loans(records, books, members):
    loans = pd.json_normalize(records)
    if "member_id" not in loans:
        loans["member_id"] = pd.NA
    if "isbn" not in loans:
        loans["isbn"] = pd.NA

    member_ids = pd.to_numeric(loans["member_id"], errors="coerce")
    _text_column(loans, "isbn")
    valid_ids = member_ids.notna() & (member_ids > 0) & (member_ids % 1 == 0)
    valid_rows = valid_ids & loans["isbn"].notna()
    missing_or_invalid = int((~valid_rows).sum())
    loans = loans.loc[valid_rows].copy()
    loans["member_id"] = member_ids.loc[valid_rows].astype("int64")

    valid_references = (
        loans["member_id"].isin(members["member_id"])
        & loans["isbn"].isin(books["isbn"])
    )
    orphan_rows = int((~valid_references).sum())
    loans = loans.loc[valid_references, LOAN_COLUMNS].copy()

    duplicate_loans = int(loans.duplicated(subset=LOAN_COLUMNS).sum())
    loans = loans.drop_duplicates(subset=LOAN_COLUMNS, keep="first")
    duplicate_book_loans = int(loans.duplicated(subset=["isbn"]).sum())
    loans = loans.drop_duplicates(subset=["isbn"], keep="first")
    return loans.reset_index(drop=True), {
        "loans_missing_or_invalid_required_fields": missing_or_invalid,
        "loans_with_unknown_book_or_member": orphan_rows,
        "duplicate_loan_records": duplicate_loans,
        "books_with_multiple_active_loan_records": duplicate_book_loans,
    }


def clean_library_data(data):
    if not isinstance(data, dict):
        raise ValueError("The root value must be an object.")

    books, book_report = _clean_books(_get_records(data, "books"))
    members, member_report = _clean_members(_get_records(data, "members"))
    loans, loan_report = _clean_loans(
        _get_records(data, "loans"),
        books,
        members,
    )

    loan_counts = loans.groupby("isbn").size()
    training_data = books.copy()
    training_data["loan_count"] = (
        training_data["isbn"].map(loan_counts).fillna(0).astype("int64")
    )
    training_data["is_available"] = np.where(
        training_data["loan_count"].to_numpy() == 0,
        True,
        False,
    )
    training_data = training_data[TRAINING_COLUMNS]

    report = {
        **book_report,
        **member_report,
        **loan_report,
        "books_exported": len(books),
        "members_exported": len(members),
        "loans_exported": len(loans),
    }
    return {
        "books": books,
        "members": members,
        "loans": loans,
        "training_data": training_data,
        "report": report,
    }


def prepare_training_data(source_path, output_dir):
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    with source_path.open("r", encoding="utf-8") as source_file:
        data = json.load(source_file)

    cleaned = clean_library_data(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("books", "members", "loans", "training_data"):
        output_path = output_dir / f"{name}.csv"
        cleaned[name].to_csv(output_path, index=False, encoding="utf-8-sig")
    return cleaned["report"]


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Clean library JSON data and export CSV tables for analysis/AI."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "library_data.json",
        help="Path to the library JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "training_data",
        help="Directory where cleaned CSV files will be written.",
    )
    args = parser.parse_args()

    report = prepare_training_data(args.input, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Clean CSV files written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
