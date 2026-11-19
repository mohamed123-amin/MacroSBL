import unittest

import pandas as pd

from project3.macroSBL import create_default_books
from prepare_training_data import clean_library_data


class CleanLibraryDataTests(unittest.TestCase):
    def test_default_books_are_cleaned_with_unique_isbns(self):
        books = pd.DataFrame(create_default_books())

        self.assertEqual(len(books), 200)
        self.assertFalse(books[["title", "author", "isbn"]].isna().any().any())
        self.assertFalse(books.duplicated(subset=["title", "author"]).any())
        self.assertFalse(books["isbn"].duplicated().any())
        self.assertTrue(books["available"].all())
        for isbn in books["isbn"]:
            weighted_sum = sum(
                int(digit) * (1 if index % 2 == 0 else 3)
                for index, digit in enumerate(isbn[:12])
            )
            self.assertEqual(int(isbn[-1]), (10 - weighted_sum % 10) % 10)

    def test_deduplicates_missing_fields_and_invalid_loans(self):
        data = {
            "books": [
                {"title": "  Book A ", "author": " Author ", "isbn": "001"},
                {"title": "Duplicate", "author": "Other", "isbn": "001"},
                {"title": "Missing author", "isbn": "002"},
                {"title": "Book B", "author": "Author B", "isbn": "002"},
            ],
            "members": [
                {"name": " Reader ", "member_id": 1},
                {"name": "Duplicate reader", "member_id": 1},
                {"name": " ", "member_id": 2},
            ],
            "loans": [
                {"member_id": 1, "isbn": "001"},
                {"member_id": 1, "isbn": "001"},
                {"member_id": 99, "isbn": "002"},
                {"member_id": 1, "isbn": "999"},
            ],
        }

        result = clean_library_data(data)

        self.assertEqual(len(result["books"]), 2)
        self.assertEqual(result["books"].iloc[0]["title"], "Book A")
        self.assertEqual(result["books"].iloc[0]["isbn"], "001")
        self.assertEqual(len(result["members"]), 1)
        self.assertEqual(len(result["loans"]), 1)
        self.assertFalse(result["training_data"].iloc[0]["is_available"])
        self.assertEqual(result["training_data"].iloc[0]["loan_count"], 1)
        self.assertTrue(result["training_data"].iloc[1]["is_available"])
        self.assertFalse(result["training_data"].isna().any().any())
        self.assertEqual(result["report"]["duplicate_book_isbns"], 1)
        self.assertEqual(result["report"]["duplicate_member_ids"], 1)
        self.assertEqual(result["report"]["duplicate_loan_records"], 1)
        self.assertEqual(result["report"]["loans_with_unknown_book_or_member"], 2)

    def test_empty_data_returns_tables_with_columns(self):
        result = clean_library_data({})

        self.assertTrue(result["books"].empty)
        self.assertTrue(result["members"].empty)
        self.assertTrue(result["loans"].empty)
        self.assertEqual(
            list(result["training_data"].columns),
            ["title", "author", "isbn", "is_available", "loan_count"],
        )
        self.assertFalse(result["training_data"].isna().any().any())

    def test_invalid_root_is_rejected(self):
        with self.assertRaises(ValueError):
            clean_library_data([])

    def test_invalid_record_collection_is_rejected(self):
        with self.assertRaises(ValueError):
            clean_library_data({"books": [None]})

    def test_missing_optional_status_is_defaulted_and_invalid_status_is_dropped(self):
        result = clean_library_data({
            "members": [
                {"name": "No status", "member_id": 1},
                {"name": "Invalid status", "member_id": 2, "active": "unknown"},
                {"name": "Inactive", "member_id": 3, "active": "no"},
            ],
        })

        self.assertEqual(len(result["members"]), 2)
        self.assertTrue(result["members"].iloc[0]["active"])
        self.assertFalse(result["members"].iloc[1]["active"])
        self.assertEqual(result["report"]["members_with_invalid_active_status"], 1)
        self.assertFalse(result["members"].isna().any().any())


if __name__ == "__main__":
    unittest.main()
