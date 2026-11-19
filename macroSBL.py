import json
import math
import os
import re
import shutil
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_BOOK_TITLES = [
    "The Atlas of Amber Skies",
    "A Clockwork Garden",
    "The Cartographer's Lantern",
    "Beneath the Quiet Mountain",
    "The Last Orchard",
    "A Map of Hidden Rivers",
    "The Glass Observatory",
    "Letters from the Blue Coast",
    "The Memory of Snow",
    "A Season of Small Wonders",
    "The City Beyond the Pines",
    "The Whispering Library",
    "An Ocean Between Worlds",
    "The Paper Moon Society",
    "Where the Wild Stars Grow",
    "The House of Falling Leaves",
    "A Bridge Made of Starlight",
    "The Keeper of Lost Maps",
    "The Clockmaker's Garden",
    "Stories from the Far Shore",
]

DEFAULT_BOOK_AUTHORS = [
    "Mira Ashford",
    "Omar Bellamy",
    "Lina Corbett",
    "Samir Daven",
    "Nora Ellery",
    "Yusuf Fairchild",
    "Talia Greer",
    "Karim Hawthorne",
    "Amira Lark",
    "Rami Vale",
]

def train_isbn_models(training_file):
    with open(training_file, "r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError("The ISBN training file must contain a 'records' list.")
    if any(not isinstance(record, dict) for record in data["records"]):
        raise ValueError("Every ISBN training record must be an object.")

    training = pd.DataFrame(data["records"])
    required_columns = {"title", "author", "sequence"}
    if not required_columns.issubset(training.columns):
        raise ValueError("Training records must contain title, author, and sequence.")
    if len(training) < 2:
        raise ValueError("At least two ISBN training records are required.")

    for column in ("title", "author"):
        if training[column].isna().any() or not training[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f"Every training record must have a non-empty {column}.")

    target = pd.to_numeric(training["sequence"], errors="coerce")
    valid_target = (
        target.notna()
        & (target > 0)
        & (target <= 999_999_999)
        & (target % 1 == 0)
    )
    if not valid_target.all():
        raise ValueError("Training sequences must be positive 9-digit integers.")

    features = training[["title", "author"]]

    def make_preprocessor():
        return ColumnTransformer(
            [
                (
                    "book_text",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["title", "author"],
                )
            ]
        )

    models = {
        "LinearRegression": make_pipeline(
            make_preprocessor(),
            LinearRegression(),
        ),
        "DecisionTreeRegressor": make_pipeline(
            make_preprocessor(),
            DecisionTreeRegressor(random_state=0),
        ),
    }

    for model in models.values():
        model.fit(features, target)

    return models


def isbn_from_sequence(sequence):
    sequence = max(1, min(999_999_999, int(round(float(sequence)))))
    isbn_base = f"978{sequence:09d}"
    total = sum(
        int(digit) * (1 if position % 2 == 0 else 3)
        for position, digit in enumerate(isbn_base)
    )
    check_digit = (10 - total % 10) % 10
    return f"{isbn_base}{check_digit}"


def predict_unique_isbn(title, author, models, used_isbns):
    book = pd.DataFrame([{"title": title, "author": author}])
    if not models:
        raise ValueError("No trained ISBN models are available.")

    predictions = [float(model.predict(book)[0]) for model in models.values()]
    if not all(math.isfinite(prediction) for prediction in predictions):
        raise ValueError("An ISBN model returned a non-finite prediction.")

    sequence = max(1, int(round(sum(predictions) / len(predictions))))
    while sequence <= 999_999_999:
        isbn = isbn_from_sequence(sequence)
        if isbn not in used_isbns:
            return isbn
        sequence += 1

    raise ValueError("Could not find an unused ISBN-shaped number.")


def create_default_books():
    books = pd.DataFrame(
        [
            {"title": title, "author": author}
            for title in DEFAULT_BOOK_TITLES
            for author in DEFAULT_BOOK_AUTHORS
        ],
        columns=["title", "author"],
    )

    for column in ("title", "author"):
        books[column] = books[column].astype("string").str.strip()
        books[column] = books[column].replace("", pd.NA)

    books = (
        books.dropna(subset=["title", "author"])
        .drop_duplicates(subset=["title", "author"], keep="first")
        .reset_index(drop=True)
    )

    isbn_numbers = np.arange(1, len(books) + 1, dtype=np.int64)
    isbn_bases = [f"978{number:09d}" for number in isbn_numbers]
    weights = np.tile(np.array([1, 3], dtype=np.int64), 6)
    check_digits = [
        str((10 - sum(int(digit) * weight for digit, weight in zip(base, weights)) % 10) % 10)
        for base in isbn_bases
    ]

    books["isbn"] = [base + check_digit for base, check_digit in zip(isbn_bases, check_digits)]
    books["available"] = np.ones(len(books), dtype=bool)
    return books.to_dict(orient="records")


class Book:
    def __init__(self, title, author, isbn, available=True, pdf_filename=None):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.available = available
        self.pdf_filename = pdf_filename

    def to_dict(self):
        return {
            "title": self.title,
            "author": self.author,
            "isbn": self.isbn,
            "available": self.available,
            "pdf_filename": self.pdf_filename,
        }


class Member:
    def __init__(self, name, member_id, active=True):
        self.name = name
        self.member_id = member_id
        self.active = active

    def to_dict(self):
        return {
            "name": self.name,
            "member_id": self.member_id,
            "active": self.active
        }


class LibraryManagementSystem:
    def __init__(self):
        self.books = []
        self.members = []
        self.loans = []
        self.data_file = Path(__file__).with_name("library_data.json")
        self.training_file = Path(__file__).with_name("isbn_training_data.json")
        self.books_folder = Path(__file__).with_name("library books")
        self.loaned_books_folder = Path(__file__).with_name("loaned books")
        self.isbn_models = None

    def ensure_books_folder(self):
        try:
            self.books_folder.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"Could not create the library books folder: {error}")
            return False
        return True

    def ensure_loaned_books_folder(self):
        try:
            self.loaned_books_folder.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"Could not create the loaned books folder: {error}")
            return False
        return True

    def load_data(self):
        if not self.ensure_books_folder():
            return False

        seed_default_books = False
        try:
            with open(self.data_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {
                "books": create_default_books(),
                "members": [],
                "loans": [],
            }
            seed_default_books = True
        except OSError as error:
            print(f"Could not read the library data file: {error}")
            return False
        except json.JSONDecodeError:
            print("Error decoding JSON data. The library was not loaded.")
            return False

        try:
            books, members, loans = self._parse_data(data)
        except ValueError as error:
            print(f"Invalid library data: {error}")
            return False

        self.books = books
        self.members = members
        self.loans = loans

        if seed_default_books or data != self._serialize_data():
            try:
                self.save_data()
            except OSError as error:
                print(f"Could not save the repaired library data: {error}")
                return False
        if seed_default_books:
            print("Created the default 200-book library dataset.")
        return True

    @staticmethod
    def _require_list(data, key):
        value = data.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"'{key}' must be a list.")
        return value

    def _parse_data(self, data):
        if not isinstance(data, dict):
            raise ValueError("The root value must be an object.")

        book_records = self._require_list(data, "books")
        member_records = self._require_list(data, "members")
        loan_records = self._require_list(data, "loans")

        books = []
        book_isbns = set()
        for record in book_records:
            if not isinstance(record, dict):
                raise ValueError("Each book must be an object.")
            title = record.get("title")
            author = record.get("author")
            isbn = record.get("isbn")
            available = record.get("available", True)
            pdf_filename = record.get("pdf_filename")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("Each book must have a non-empty title.")
            if not isinstance(author, str) or not author.strip():
                raise ValueError("Each book must have a non-empty author.")
            if not isinstance(isbn, str) or not isbn.isdigit():
                raise ValueError("Each book must have a numeric ISBN.")
            if isbn in book_isbns:
                raise ValueError(f"Duplicate book ISBN: {isbn}.")
            if not isinstance(available, bool):
                raise ValueError(f"Availability for ISBN {isbn} must be boolean.")
            if pdf_filename is not None and not isinstance(pdf_filename, str):
                raise ValueError(f"PDF filename for ISBN {isbn} must be text.")
            if pdf_filename is not None:
                self.get_pdf_path(pdf_filename)

            books.append(Book(title, author, isbn, available, pdf_filename))
            book_isbns.add(isbn)

        members = []
        member_ids = set()
        for record in member_records:
            if not isinstance(record, dict):
                raise ValueError("Each member must be an object.")
            name = record.get("name")
            member_id = record.get("member_id")
            active = record.get("active", True)
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each member must have a non-empty name.")
            if isinstance(member_id, bool) or not isinstance(member_id, int) or member_id <= 0:
                raise ValueError("Each member must have a positive integer ID.")
            if member_id in member_ids:
                raise ValueError(f"Duplicate member ID: {member_id}.")
            if not isinstance(active, bool):
                raise ValueError(f"Active status for member {member_id} must be boolean.")

            members.append(Member(name, member_id, active))
            member_ids.add(member_id)

        loans = []
        loaned_isbns = set()
        loan_keys = set()
        for record in loan_records:
            if not isinstance(record, dict):
                raise ValueError("Each loan must be an object.")
            member_id = record.get("member_id")
            isbn = record.get("isbn")
            if isinstance(member_id, bool) or not isinstance(member_id, int) or member_id <= 0:
                raise ValueError("Each loan must reference a positive integer member ID.")
            if not isinstance(isbn, str) or not isbn.isdigit():
                raise ValueError("Each loan must reference a numeric ISBN.")
            if member_id not in member_ids:
                raise ValueError(f"Loan references missing member ID {member_id}.")
            if isbn not in book_isbns:
                raise ValueError(f"Loan references missing book ISBN {isbn}.")
            if isbn in loaned_isbns:
                raise ValueError(f"Book ISBN {isbn} has more than one active loan.")
            key = (member_id, isbn)
            if key in loan_keys:
                raise ValueError(f"Duplicate loan for member {member_id} and ISBN {isbn}.")

            loans.append({"member_id": member_id, "isbn": isbn})
            loaned_isbns.add(isbn)
            loan_keys.add(key)

        for book in books:
            book.available = book.isbn not in loaned_isbns

        return books, members, loans

    def _serialize_data(self):
        return {
            "books": [book.to_dict() for book in self.books],
            "members": [member.to_dict() for member in self.members],
            "loans": self.loans,
        }

    def save_data(self):
        data = self._serialize_data()
        temporary_file = self.data_file.with_suffix(".tmp")
        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        temporary_file.replace(self.data_file)

    def _save_or_report(self, action):
        try:
            self.save_data()
        except OSError as error:
            print(f"Could not {action}: {error}")
            return False
        return True

    @staticmethod
    def make_pdf_filename(title, isbn):
        safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)
        safe_title = safe_title.strip(" .")[:150] or "book"
        return f"{isbn} - {safe_title}.pdf"

    def get_pdf_path(self, filename):
        if not filename or Path(filename).name != filename or Path(filename).suffix.lower() != ".pdf":
            raise ValueError("The saved PDF filename is invalid.")
        library_path = self.books_folder / filename
        loaned_path = self.loaned_books_folder / filename
        if library_path.is_file():
            return library_path
        if loaned_path.is_file():
            return loaned_path
        return library_path

    @staticmethod
    def _is_path_within(path, root):
        path_text = os.path.normcase(os.path.abspath(path))
        root_text = os.path.normcase(os.path.abspath(root))
        try:
            return os.path.commonpath([path_text, root_text]) == root_text
        except ValueError:
            return False

    @staticmethod
    def _normalize_isbn_candidate(candidate):
        isbn = re.sub(r"[^0-9X]", "", candidate.upper())
        if len(isbn) == 13 and isbn[:3] in {"978", "979"}:
            weighted_sum = sum(
                int(digit) * (1 if position % 2 == 0 else 3)
                for position, digit in enumerate(isbn[:12])
            )
            check_digit = (10 - weighted_sum % 10) % 10
            return isbn if int(isbn[-1]) == check_digit else None

        if len(isbn) == 10:
            weighted_sum = sum(
                int(digit) * (10 - position)
                for position, digit in enumerate(isbn[:9])
            )
            check_digit = 10 if isbn[-1] == "X" else int(isbn[-1])
            if (weighted_sum + check_digit) % 11 != 0:
                return None

            isbn13_base = f"978{isbn[:9]}"
            isbn13_sum = sum(
                int(digit) * (1 if position % 2 == 0 else 3)
                for position, digit in enumerate(isbn13_base)
            )
            return isbn13_base + str((10 - isbn13_sum % 10) % 10)

        return None

    @classmethod
    def _extract_pdf_catalog_details(cls, pdf_path):
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        title = None
        author = None
        try:
            reader = PdfReader(str(pdf_path), strict=False)
            metadata = reader.metadata or {}
            title_value = metadata.get("/Title")
            author_value = metadata.get("/Author")
            title = title_value.strip() if isinstance(title_value, str) else None
            author = author_value.strip() if isinstance(author_value, str) else None
            title = title or None
            author = author or None

            isbn = None
            candidate_pattern = (
                r"(?<![0-9A-Z])(?:97[89][ -]?)?"
                r"(?:[0-9][ -]?){9}[0-9X](?![0-9A-Z])"
            )
            for page in reader.pages:
                text = page.extract_text() or ""
                for candidate in re.findall(candidate_pattern, text, re.IGNORECASE):
                    isbn = cls._normalize_isbn_candidate(candidate)
                    if isbn:
                        break
                if isbn:
                    break
            return {"title": title, "author": author, "isbn": isbn}
        except (OSError, PdfReadError, ValueError) as error:
            return {
                "title": title,
                "author": author,
                "isbn": None,
                "error": str(error),
            }

    def _register_copied_pdf(self, title, author, isbn, pdf_filename):
        title = title.strip() if isinstance(title, str) else ""
        author = author.strip() if isinstance(author, str) else ""
        if not title or not author:
            raise ValueError("A book title and author are required for registration.")
        if not isinstance(isbn, str) or not isbn.isdigit():
            raise ValueError("A valid numeric ISBN is required for registration.")
        if Path(pdf_filename).name != pdf_filename:
            raise ValueError("The saved PDF filename is invalid.")

        pdf_path = self.books_folder / pdf_filename
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found in the library: {pdf_path}")
        if any(book.isbn == isbn for book in self.books):
            raise ValueError(f"ISBN '{isbn}' is already registered.")
        if any(book.pdf_filename == pdf_filename for book in self.books):
            raise ValueError(f"PDF '{pdf_filename}' is already registered.")

        book = Book(title, author, isbn, pdf_filename=pdf_filename)
        self.books.append(book)
        try:
            self.save_data()
        except OSError:
            self.books.remove(book)
            raise
        return book.to_dict()

    def find_and_copy_pdf(
        self,
        search_term=None,
        pdf_path=None,
        search_root=None,
        title=None,
        author=None,
    ):
        for field_name, value in (("title", title), ("author", author)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"The book {field_name} must be text.")

        if pdf_path is not None:
            if not isinstance(pdf_path, str) or not pdf_path.strip():
                raise ValueError("Provide a valid PDF path.")
            if search_term is not None:
                if not isinstance(search_term, str):
                    raise ValueError("The PDF filename search term must be text.")
                if search_term.strip():
                    raise ValueError(
                        "Provide either a PDF path or a filename search term."
                    )
            source = Path(pdf_path).expanduser().resolve()
            profile_root = Path.home().resolve()
            if not self._is_path_within(source, profile_root):
                raise ValueError("The PDF path must be inside the current user profile.")
            if source.suffix.lower() != ".pdf":
                raise ValueError("Only PDF files can be copied into the library.")
            if not source.is_file():
                raise FileNotFoundError(f"PDF file not found: {source}")
            matches = [source]
            scan_errors = []
        else:
            if not isinstance(search_term, str) or not search_term.strip():
                raise ValueError("Provide a PDF path or part of its filename.")
            search_term = search_term.strip()
            root = Path(search_root or Path.home()).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(f"Search folder not found: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"Search path is not a folder: {root}")

            excluded_names = {
                "appdata",
                "application data",
                "$recycle.bin",
                "system volume information",
                ".git",
                ".venv",
                "venv",
                "node_modules",
                "__pycache__",
                ".cache",
            }
            excluded_roots = [
                self.books_folder.resolve(),
                self.loaned_books_folder.resolve(),
            ]
            is_junction = getattr(os.path, "isjunction", None)
            matches = []
            scan_errors = []

            def record_walk_error(error):
                scan_errors.append(
                    {"path": error.filename, "error": str(error)}
                )

            for current, directories, filenames in os.walk(
                root,
                topdown=True,
                onerror=record_walk_error,
                followlinks=False,
            ):
                current_path = Path(current)
                allowed_directories = []
                for name in directories:
                    candidate = current_path / name
                    if (
                        name.casefold() in excluded_names
                        or name.startswith(".")
                        or os.path.islink(candidate)
                        or (is_junction is not None and is_junction(candidate))
                        or any(
                            self._is_path_within(candidate, excluded_root)
                            for excluded_root in excluded_roots
                        )
                    ):
                        continue
                    allowed_directories.append(name)
                directories[:] = allowed_directories

                for filename in filenames:
                    candidate = current_path / filename
                    if (
                        candidate.suffix.lower() == ".pdf"
                        and search_term.casefold() in filename.casefold()
                        and not os.path.islink(candidate)
                        and candidate.is_file()
                    ):
                        resolved_candidate = candidate.resolve()
                        if not self._is_path_within(resolved_candidate, root):
                            continue
                        matches.append(resolved_candidate)
                        if len(matches) == 2:
                            break
                if len(matches) == 2:
                    break

        if not matches:
            return {
                "found": False,
                "copied": False,
                "matches": [],
                "message": "No matching PDF was found in the searched folders.",
                "warnings": scan_errors,
            }
        if len(matches) > 1:
            return {
                "found": True,
                "copied": False,
                "matches": [str(path) for path in matches],
                "message": "More than one PDF matched; use a more specific filename.",
                "warnings": scan_errors,
            }

        source = matches[0]
        if self._is_path_within(source, self.books_folder.resolve()):
            book = next(
                (book for book in self.books if book.pdf_filename == source.name),
                None,
            )
            return {
                "found": True,
                "copied": False,
                "source": str(source),
                "isbn": book.isbn if book else None,
                "cataloged": book is not None,
                "book": book.to_dict() if book else None,
                "message": (
                    "This PDF is already in the library books folder."
                    if book
                    else "This PDF is already in the library but is not registered."
                ),
                "warnings": scan_errors,
            }

        details = self._extract_pdf_catalog_details(source)
        warnings = list(scan_errors)
        if details.get("error"):
            warnings.append({"path": str(source), "error": details["error"]})

        isbn = details.get("isbn")
        existing_book = next(
            (book for book in self.books if book.isbn == isbn),
            None,
        )
        if existing_book:
            return {
                "found": True,
                "copied": False,
                "source": str(source),
                "isbn": isbn,
                "cataloged": True,
                "book": existing_book.to_dict(),
                "message": (
                    f"ISBN '{isbn}' is already registered; no duplicate was added."
                ),
                "warnings": warnings,
            }

        book_title = title.strip() if title and title.strip() else details["title"]
        book_author = author.strip() if author and author.strip() else details["author"]

        self.books_folder.mkdir(parents=True, exist_ok=True)
        destination = self.books_folder / source.name
        if destination.exists():
            existing_book = next(
                (
                    book
                    for book in self.books
                    if book.pdf_filename == destination.name
                ),
                None,
            )
            return {
                "found": True,
                "copied": False,
                "source": str(source),
                "destination": str(destination),
                "message": "A PDF with this filename already exists in the library.",
                "isbn": existing_book.isbn if existing_book else isbn,
                "cataloged": existing_book is not None,
                "book": existing_book.to_dict() if existing_book else None,
                "warnings": warnings,
            }

        destination_created = False
        book_record = None
        try:
            with source.open("rb") as source_file:
                if source_file.read(5) != b"%PDF-":
                    raise ValueError(f"The selected file is not a valid PDF: {source}")
                source_file.seek(0)
                with destination.open("xb") as destination_file:
                    destination_created = True
                    shutil.copyfileobj(source_file, destination_file)
            shutil.copystat(source, destination)
            if isbn and book_title and book_author:
                book_record = self._register_copied_pdf(
                    book_title,
                    book_author,
                    isbn,
                    destination.name,
                )
        except FileExistsError:
            return {
                "found": True,
                "copied": False,
                "source": str(source),
                "destination": str(destination),
                "message": "A PDF with this filename already exists in the library.",
                "warnings": warnings,
            }
        except (OSError, ValueError):
            if destination_created and destination.exists():
                try:
                    destination.unlink()
                except OSError as cleanup_error:
                    raise OSError(
                        f"PDF copy failed and the partial destination could not be "
                        f"removed: {destination}: {cleanup_error}"
                    ) from cleanup_error
            raise

        return {
            "found": True,
            "copied": True,
            "source": str(source),
            "destination": str(destination),
            "isbn": isbn,
            "cataloged": book_record is not None,
            "book": book_record,
            "title": book_title,
            "author": book_author,
            "needs_book_details": (
                [
                    field
                    for field, value in (
                        ("title", book_title),
                        ("author", book_author),
                    )
                    if not value
                ]
                if isbn and book_record is None
                else []
            ),
            "message": (
                f"PDF copied and registered with ISBN '{isbn}'."
                if book_record
                else (
                    f"PDF copied; ISBN '{isbn}' was found. "
                    "Book details are needed before catalog registration."
                    if isbn
                    else "PDF copied, but no valid ISBN was found; it was not added to the catalog."
                )
            ),
            "warnings": warnings,
        }

    def add_book(self, title=None, author=None, pdf_path=None):
        title = (title if title is not None else input("Enter the title of the book: ")).strip()
        author = (author if author is not None else input("Enter the author of the book: ")).strip()
        pdf_path_input = (
            pdf_path if pdf_path is not None else input("Enter the path to the PDF file: ")
        ).strip()
        if (
            len(pdf_path_input) >= 2
            and pdf_path_input[0] == pdf_path_input[-1]
            and pdf_path_input[0] in {"'", '"'}
        ):
            pdf_path_input = pdf_path_input[1:-1].strip()
        pdf_source = Path(pdf_path_input).expanduser()

        if title == "" or author == "":
            print("Title and author cannot be empty. Please provide valid inputs.")
            return
        if pdf_source.suffix.lower() != ".pdf":
            print("Only PDF files can be added.")
            return
        if not pdf_source.is_file():
            print(f"PDF file not found: {pdf_source}")
            return
        try:
            with pdf_source.open("rb") as source_file:
                if source_file.read(5) != b"%PDF-":
                    print("The selected file does not contain a valid PDF header.")
                    return
        except OSError as error:
            print(f"Could not read the selected PDF file: {error}")
            return

        if not self.ensure_books_folder():
            return

        used_isbns = {book.isbn for book in self.books}
        try:
            if self.isbn_models is None:
                self.isbn_models = train_isbn_models(self.training_file)
            isbn = predict_unique_isbn(
                title,
                author,
                self.isbn_models,
                used_isbns,
            )
        except (OSError, ValueError) as error:
            print(f"Could not generate an ISBN-shaped number: {error}")
            return

        pdf_filename = self.make_pdf_filename(title, isbn)
        pdf_destination = self.books_folder / pdf_filename
        if pdf_destination.exists():
            print(f"A PDF with this destination name already exists: {pdf_destination}")
            return

        book = None
        try:
            shutil.copy2(pdf_source, pdf_destination)
            book = Book(title, author, isbn, pdf_filename=pdf_filename)
            self.books.append(book)
            self.save_data()
        except OSError as error:
            if book in self.books:
                self.books.remove(book)
            if pdf_destination.exists():
                try:
                    pdf_destination.unlink()
                except OSError as cleanup_error:
                    print(f"Could not remove the copied PDF after the failed add: {cleanup_error}")
            print(f"Could not add the PDF book: {error}")
            return
        print(f"Book '{title}' added successfully!")
        print(f"Your internal ISBN-shaped number is {isbn}")
        return book.to_dict()

    def remove_book(self, isbn=None):
        isbn = isbn if isbn is not None else input("Enter the ISBN of the book to remove: ")

        for book in self.books:
            if book.isbn == isbn:
                for loan in self.loans:
                    if loan["isbn"] == isbn:
                        print(f"Book with ISBN '{isbn}' is currently loaned out and cannot be removed.")
                        return

                if not self.ensure_books_folder():
                    return

                pdf_path = None
                staged_pdf = None
                book_index = self.books.index(book)
                if book.pdf_filename:
                    try:
                        pdf_path = self.get_pdf_path(book.pdf_filename)
                    except ValueError as error:
                        print(error)
                        return
                    if not pdf_path.is_file():
                        print(f"PDF file is missing; book was not removed: {pdf_path}")
                        return
                    staged_pdf = pdf_path.with_name(f"{pdf_path.name}.deleting")
                    if staged_pdf.exists():
                        print(f"Cannot safely remove the book because this file already exists: {staged_pdf}")
                        return

                try:
                    if pdf_path and staged_pdf:
                        pdf_path.replace(staged_pdf)
                    self.books.remove(book)
                    self.save_data()
                except OSError as error:
                    if book not in self.books:
                        self.books.insert(book_index, book)
                    if staged_pdf and staged_pdf.exists() and pdf_path:
                        try:
                            staged_pdf.replace(pdf_path)
                        except OSError as restore_error:
                            print(f"Could not restore the book PDF: {restore_error}")
                    print(f"Could not remove the book: {error}")
                    return

                if staged_pdf and staged_pdf.exists():
                    try:
                        staged_pdf.unlink()
                    except OSError as error:
                        print(f"Book record removed, but could not delete its PDF: {error}")
                        return
                print(f"Book with ISBN '{isbn}' and its PDF removed successfully!")
                return {"isbn": isbn, "removed": True}

        print(f"No book found with ISBN '{isbn}'.")

    def modify_book(self, isbn=None, new_title=None, new_author=None):
        isbn = isbn if isbn is not None else input("Enter the ISBN of the book to modify: ")

        for book in self.books:
            if book.isbn == isbn:
                new_title = (
                    new_title
                    if new_title is not None
                    else input("Enter the new title of the book: ")
                )
                new_author = (
                    new_author
                    if new_author is not None
                    else input("Enter the new author of the book: ")
                )
                new_title = new_title.strip()
                new_author = new_author.strip()

                if new_title.strip() == "" or new_author.strip() == "":
                    print("Title and author cannot be empty. Please provide valid inputs.")
                    return

                if not self.ensure_books_folder():
                    return

                old_title = book.title
                old_author = book.author
                old_filename = book.pdf_filename
                new_pdf_path = None
                old_pdf_path = None
                if old_filename:
                    try:
                        old_pdf_path = self.get_pdf_path(old_filename)
                    except ValueError as error:
                        print(error)
                        return
                    if not old_pdf_path.is_file():
                        print(f"PDF file is missing; book was not modified: {old_pdf_path}")
                        return
                    new_filename = self.make_pdf_filename(new_title, isbn)
                    new_pdf_path = old_pdf_path.with_name(new_filename)
                    other_folder = (
                        self.loaned_books_folder
                        if old_pdf_path.parent == self.books_folder
                        else self.books_folder
                    )
                    other_pdf_path = other_folder / new_filename
                    if new_pdf_path != old_pdf_path and (
                        new_pdf_path.exists() or other_pdf_path.exists()
                    ):
                        print(f"A PDF with the new title already exists: {new_pdf_path}")
                        return

                try:
                    if old_pdf_path and new_pdf_path and new_pdf_path != old_pdf_path:
                        old_pdf_path.replace(new_pdf_path)
                        book.pdf_filename = new_pdf_path.name
                    book.title = new_title
                    book.author = new_author
                    self.save_data()
                except OSError as error:
                    book.title = old_title
                    book.author = old_author
                    book.pdf_filename = old_filename
                    if new_pdf_path and old_pdf_path and new_pdf_path.exists():
                        try:
                            new_pdf_path.replace(old_pdf_path)
                        except OSError as restore_error:
                            print(f"Could not restore the original book PDF: {restore_error}")
                    print(f"Could not modify the book: {error}")
                    return

                print(f"Book with ISBN '{isbn}' modified successfully!")
                return book.to_dict()

        print(f"No book found with ISBN '{isbn}'.")

    def search_book(self, search_term=None):
        search_term = (
            search_term
            if search_term is not None
            else input("Enter the title or author of the book to search: ")
        )

        if search_term.strip() == "":
            print("Search term cannot be empty. Please provide a valid input.")
            return

        found_books = []
        for book in self.books:
            if search_term.lower() in book.title.lower() or search_term.lower() in book.author.lower():
                found_books.append(book)

        if found_books:
            print("Found books:")
            for book in found_books:
                print(f"Title: {book.title}, Author: {book.author}, ISBN: {book.isbn}")
            return [book.to_dict() for book in found_books]
        else:
            print("No books found matching the search term.")
            return []

    def see_all_books(self):
        if self.books:
            print("All books in the library:")
            for book in self.books:
                print(f"Title: {book.title}, Author: {book.author}, ISBN: {book.isbn}")
        else:
            print("No books available in the library.")
        return [book.to_dict() for book in self.books]

    def add_member(self, name=None):
        name = (name if name is not None else input("Enter the name of the member: ")).strip()

        if name == "":
            print("Member name cannot be empty. Please provide a valid name.")
            return

        member_id = 0
        for member in self.members:
            if member.member_id > member_id:
                member_id = member.member_id
        member_id += 1

        member = Member(name, member_id)
        self.members.append(member)
        if not self._save_or_report("add the member"):
            self.members.remove(member)
            return
        print(f"Member '{name}' added successfully!")
        print(f"Member ID: {member_id}")
        return member.to_dict()

    def remove_member(self, member_id=None):
        try:
            member_id = int(
                member_id
                if member_id is not None
                else input("Enter the member ID to remove: ")
            )
        except ValueError:
            print("Invalid member ID. Please enter a valid integer.")
            return

        for member in self.members:
            if member.member_id == member_id:
                for loan in self.loans:
                    if loan["member_id"] == member_id:
                        print(f"Member with ID '{member_id}' has active loans and cannot be removed.")
                        return

                member_index = self.members.index(member)
                self.members.remove(member)
                if not self._save_or_report("remove the member"):
                    self.members.insert(member_index, member)
                    return
                print(f"Member with ID '{member_id}' removed successfully!")
                return {"member_id": member_id, "removed": True}

        print(f"No member found with ID '{member_id}'.")

    def loan_book(self, member_id=None, isbn=None):
        try:
            member_id = int(
                member_id if member_id is not None else input("Enter the member ID: ")
            )
            if member_id <= 0:
                print("Member ID should be a positive integer. Please provide a valid member ID.")
                return
        except ValueError:
            print("Invalid member ID. Please enter a valid integer.")
            return

        isbn = isbn if isbn is not None else input("Enter the ISBN of the book to loan: ")
        if isbn.strip() == "":
            print("ISBN cannot be empty. Please provide a valid ISBN.")
            return
        if not isbn.isdigit():
            print("ISBN should be a numeric value. Please provide a valid ISBN.")
            return

        member = None
        for current_member in self.members:
            if current_member.member_id == member_id:
                member = current_member
                break

        if member is None:
            print(f"No member found with ID '{member_id}'.")
            return
        if not member.active:
            print(f"Member with ID '{member_id}' is not active and cannot loan books.")
            return

        book = None
        for current_book in self.books:
            if current_book.isbn == isbn:
                book = current_book
                break

        if book is None:
            print(f"No book found with ISBN '{isbn}'.")
            return
        if any(loan["isbn"] == isbn for loan in self.loans):
            print(f"Book with ISBN '{isbn}' is already loaned out.")
            return
        if not book.available:
            print(f"Book with ISBN '{isbn}' is not available.")
            return

        pdf_source = None
        pdf_destination = None
        pdf_moved = False
        if book.pdf_filename:
            try:
                pdf_source = self.get_pdf_path(book.pdf_filename)
            except ValueError as error:
                print(error)
                return
            if not pdf_source.is_file():
                print(f"Book PDF is missing; the loan was not created: {pdf_source}")
                return

            pdf_destination = self.loaned_books_folder / book.pdf_filename
            if pdf_source != pdf_destination:
                if pdf_destination.exists():
                    print(
                        "A PDF with this name already exists in the loaned books "
                        f"folder: {pdf_destination}"
                    )
                    return
                if not self.ensure_loaned_books_folder():
                    return
                try:
                    pdf_source.replace(pdf_destination)
                except OSError as error:
                    print(f"Could not move the book PDF to the loaned books folder: {error}")
                    return
                pdf_moved = True

        loan = {
            "member_id": member_id,
            "isbn": isbn
        }
        self.loans.append(loan)
        book.available = False
        if not self._save_or_report("loan the book"):
            self.loans.remove(loan)
            book.available = True
            if pdf_moved and pdf_destination and pdf_source:
                try:
                    pdf_destination.replace(pdf_source)
                except OSError as error:
                    print(f"Could not restore the book PDF after the failed loan: {error}")
            return
        print(f"Book with ISBN '{isbn}' loaned to member ID '{member_id}' successfully!")
        return {"member_id": member_id, "isbn": isbn, "loaned": True}

    def see_all_loans(self):
        if self.loans:
            print("All loans in the library:")
            for loan in self.loans:
                print(f"Member ID: {loan['member_id']}, ISBN: {loan['isbn']}")
        else:
            print("No loans available in the library.")
        return list(self.loans)

    def see_all_members(self):
        if self.members:
            print("All members in the library:")
            for member in self.members:
                print(f"Name: {member.name}, Member ID: {member.member_id}, Active: {member.active}")
        else:
            print("No members available in the library.")
        return [member.to_dict() for member in self.members]

    def return_book(self, member_id=None, isbn=None):
        try:
            member_id = int(
                member_id if member_id is not None else input("Enter the member ID: ")
            )
            if member_id <= 0:
                print("Member ID should be a positive integer. Please provide a valid member ID.")
                return
        except ValueError:
            print("Invalid member ID. Please enter a valid integer.")
            return

        isbn = isbn if isbn is not None else input("Enter the ISBN of the book to return: ")
        if isbn.strip() == "":
            print("ISBN cannot be empty. Please provide a valid ISBN.")
            return
        if not isbn.isdigit():
            print("ISBN should be a numeric value. Please provide a valid ISBN.")
            return

        loan = next(
            (
                current_loan
                for current_loan in self.loans
                if current_loan["member_id"] == member_id and current_loan["isbn"] == isbn
            ),
            None,
        )
        if loan is not None:
            book = next((current_book for current_book in self.books if current_book.isbn == isbn), None)
            if book is None:
                print(f"Cannot return the book: ISBN '{isbn}' is missing from the library.")
                return

            pdf_source = None
            pdf_destination = None
            pdf_moved = False
            pdf_missing = False
            if book.pdf_filename:
                try:
                    self.get_pdf_path(book.pdf_filename)
                except ValueError as error:
                    print(error)
                    return

                library_pdf = self.books_folder / book.pdf_filename
                loaned_pdf = self.loaned_books_folder / book.pdf_filename
                if loaned_pdf.is_file():
                    if library_pdf.exists():
                        print(
                            "Cannot return the book PDF because a file with the same "
                            f"name already exists in library books: {library_pdf}"
                        )
                        return
                    if not self.ensure_books_folder():
                        return
                    pdf_source = loaned_pdf
                    pdf_destination = library_pdf
                    try:
                        pdf_source.replace(pdf_destination)
                    except OSError as error:
                        print(f"Could not move the book PDF back to library books: {error}")
                        return
                    pdf_moved = True
                elif not library_pdf.is_file():
                    pdf_missing = True

            old_available = book.available
            self.loans.remove(loan)
            book.available = not any(current_loan["isbn"] == isbn for current_loan in self.loans)
            if not self._save_or_report("return the book"):
                self.loans.append(loan)
                book.available = old_available
                if pdf_moved and pdf_destination and pdf_source:
                    try:
                        pdf_destination.replace(pdf_source)
                    except OSError as error:
                        print(f"Could not restore the loaned PDF after the failed return: {error}")
                return
            if pdf_missing:
                print("The book was returned, but its saved PDF is missing.")
            print(f"Book with ISBN '{isbn}' returned by member ID '{member_id}' successfully!")
            return {"member_id": member_id, "isbn": isbn, "returned": True}

        print(f"No loan found for member ID '{member_id}' with ISBN '{isbn}'.")

    def plot_book_availability(self):
        if not self.books:
            print("No books available to display.")
            return

        available = sum(book.available for book in self.books)
        on_loan = len(self.books) - available

        labels = ["Available", "On loan"]
        counts = [available, on_loan]

        fig, ax = plt.subplots()
        bars = ax.bar(labels, counts, color=["seagreen", "tomato"])
        ax.set_title("Library book availability")
        ax.set_ylabel("Number of books")
        ax.bar_label(bars)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    def run(self):
        if not self.load_data():
            return

        while True:
            print("Welcome to MacroSBL!")
            print("===============================")
            print("Please select an option:")
            print("1. Add a book")
            print("2. Remove a book")
            print("3. Modify a book")
            print("4. Search for a book")
            print("5. See all books")
            print("6. Add a member")
            print("7. Remove a member")
            print("8. Loan a book")
            print("9. See all loans")
            print("10. Return a book")
            print("11. Show book availability chart")
            print("12. Exit")
            print("13. Use the local tools agent")
            print("===============================")

            try:
                option = int(input("Enter your option as a number: "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            if option < 1 or option > 13:
                print("Invalid option. Please select a number between 1 and 13.")
            elif option == 1:
                self.add_book()
            elif option == 2:
                self.remove_book()
            elif option == 3:
                self.modify_book()
            elif option == 4:
                self.search_book()
            elif option == 5:
                self.see_all_books()
            elif option == 6:
                self.add_member()
            elif option == 7:
                self.remove_member()
            elif option == 8:
                self.loan_book()
            elif option == 9:
                self.see_all_loans()
            elif option == 10:
                self.return_book()
            elif option == 11:
                self.plot_book_availability()
            elif option == 12:
                print("Exiting MacroSBL. farewell!")
                break
            else:
                if __package__:
                    from .library_agent import LocalLibraryAgent
                else:
                    from library_agent import LocalLibraryAgent
                if LocalLibraryAgent(self).run():
                    print("Exiting MacroSBL. farewell!")
                    break
            
if __name__ == "__main__":
    library = LibraryManagementSystem()
    library.run()