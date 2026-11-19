import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project3.macroSBL import (
    Book,
    LibraryManagementSystem,
    isbn_from_sequence,
    predict_unique_isbn,
    train_isbn_models,
)


class ISBNModelTests(unittest.TestCase):
    def setUp(self):
        self.training_file = Path(__file__).with_name("isbn_training_data.json")
        self.models = train_isbn_models(self.training_file)

    def test_trains_both_models_from_training_file(self):
        self.assertEqual(
            set(self.models),
            {"LinearRegression", "DecisionTreeRegressor"},
        )

    def test_prediction_has_valid_check_digit_and_is_unique(self):
        training_records = json.loads(
            self.training_file.read_text(encoding="utf-8-sig")
        )["records"]
        used_isbns = {record["isbn"] for record in training_records}
        predicted = predict_unique_isbn(
            "A Completely New Book",
            "A New Author",
            self.models,
            used_isbns,
        )
        self.assertEqual(len(predicted), 13)
        self.assertTrue(predicted.isdigit())
        self.assertEqual(predicted[:3], "978")
        self.assertEqual(predicted, isbn_from_sequence(int(predicted[3:12])))
        self.assertNotIn(predicted, used_isbns)

        next_prediction = predict_unique_isbn(
            "A Completely New Book",
            "A New Author",
            self.models,
            used_isbns | {predicted},
        )
        self.assertNotEqual(next_prediction, predicted)

    def test_prediction_combines_both_model_outputs(self):
        class FixedModel:
            def __init__(self, value):
                self.value = value

            def predict(self, _book):
                return [self.value]

        predicted = predict_unique_isbn(
            "Book",
            "Author",
            {
                "LinearRegression": FixedModel(10),
                "DecisionTreeRegressor": FixedModel(20),
            },
            set(),
        )

        self.assertEqual(predicted, isbn_from_sequence(15))

    def test_add_book_saves_predicted_isbn_and_copies_pdf(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_pdf = temporary_path / "source.pdf"
            source_pdf.write_bytes(b"%PDF-1.4\n")

            library = LibraryManagementSystem()
            library.data_file = temporary_path / "library_data.json"
            library.training_file = self.training_file
            library.books_folder = temporary_path / "library books"
            training_records = json.loads(
                self.training_file.read_text(encoding="utf-8-sig")
            )["records"]
            library.books = [
                Book(record["title"], record["author"], record["isbn"])
                for record in training_records
            ]

            with patch(
                "builtins.input",
                side_effect=[
                    "New Test Book",
                    "Test Author",
                    str(source_pdf),
                ],
            ), contextlib.redirect_stdout(io.StringIO()):
                library.add_book()

            self.assertEqual(len(library.books), 201)
            added_book = library.books[-1]
            self.assertNotIn(
                added_book.isbn,
                {record["isbn"] for record in training_records},
            )
            self.assertEqual(len(added_book.isbn), 13)
            self.assertEqual(
                added_book.isbn,
                isbn_from_sequence(int(added_book.isbn[3:12])),
            )

            pdf_path = library.books_folder / added_book.pdf_filename
            self.assertTrue(pdf_path.is_file())
            self.assertEqual(pdf_path.read_bytes(), source_pdf.read_bytes())

            saved_data = json.loads(library.data_file.read_text(encoding="utf-8"))
            self.assertEqual(saved_data["books"][-1]["isbn"], added_book.isbn)
            self.assertEqual(
                saved_data["books"][-1]["pdf_filename"],
                added_book.pdf_filename,
            )


if __name__ == "__main__":
    unittest.main()
