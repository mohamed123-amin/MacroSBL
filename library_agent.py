import json
import re
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

if __package__:
    from .sparse_clustering import cluster_sparse_vectors
    from .self_learning import SelfLearningMemory
    from .text_generator import LocalTextGenerator
else:
    from sparse_clustering import cluster_sparse_vectors
    from self_learning import SelfLearningMemory
    from text_generator import LocalTextGenerator


INTENT_EXAMPLES = {
    "add_book": [
        "add a book",
        "please add this book to the library",
        "اضف كتاب",
        "أضف كتابا جديدا للمكتبة",
        "عايز اسجل كتاب جديد",
    ],
    "remove_book": [
        "remove a book",
        "delete the book with this ISBN",
        "احذف الكتاب",
        "امسح كتاب من المكتبة",
        "عايز اشيل الكتاب ده",
    ],
    "modify_book": [
        "modify a book",
        "change the book details",
        "عدل بيانات الكتاب",
        "غير عنوان الكتاب",
        "عايز اعدل كتاب",
    ],
    "search_book": [
        "search for a book",
        "find books by title or author",
        "ابحث عن كتاب",
        "دور على كتاب",
        "عايز الاقي الكتاب",
    ],
    "find_and_copy_pdf": [
        "find a PDF in my folders and copy it to the library",
        "search my folders for a PDF file",
        "locate a PDF by filename",
        "copy this PDF from its path into the library",
        "ابحث عن ملف PDF في المجلدات وانسخه للمكتبة",
        "دور على ملف PDF وانسخه إلى المكتبة",
    ],
    "see_all_books": [
        "show all books",
        "list every book in the library",
        "اعرض كل الكتب",
        "وريني الكتب الموجودة",
        "عايز اشوف كل الكتب",
    ],
    "add_member": [
        "add a library member",
        "register a new member",
        "اضف عضو",
        "سجل عضو جديد",
        "عايز اضيف مشترك",
    ],
    "remove_member": [
        "remove a member",
        "delete this library member",
        "احذف العضو",
        "امسح عضو",
        "عايز اشيل مشترك",
    ],
    "see_all_members": [
        "show all members",
        "list library members",
        "اعرض كل الاعضاء",
        "وريني المشتركين",
        "عايز اشوف الاعضاء",
    ],
    "loan_book": [
        "loan a book to a member",
        "check out this book",
        "اعير كتاب لعضو",
        "سجل استعارة كتاب",
        "عايز اعير كتاب",
    ],
    "see_all_loans": [
        "show all loans",
        "list borrowed books",
        "اعرض كل الاستعارات",
        "وريني الكتب المستعارة",
        "عايز اشوف الاستعارات",
    ],
    "return_book": [
        "return a borrowed book",
        "check in this book",
        "رجع الكتاب",
        "سجل ارجاع الكتاب",
        "عايز ارجع كتاب",
    ],
    "read_pdf": [
        "read this PDF",
        "extract the text from the PDF",
        "اقرأ ملف pdf",
        "استخرج النص من الملف",
        "عايز اقرا ملف بي دي اف",
    ],
    "analyze_pdf": [
        "analyze this PDF",
        "extract information and ISBN from the PDF",
        "حلل ملف pdf",
        "استخرج بيانات الملف",
        "عايز احلل الكتاب",
    ],
    "learn_pdf": [
        "learn the information in this PDF",
        "add this PDF to the local knowledge base",
        "تعلم المعلومات من ملف pdf",
        "احفظ محتوى الملف للبحث",
        "ضيف الكتاب لقاعدة المعرفة",
    ],
    "ocr_pdf": [
        "run OCR on this scanned PDF",
        "recognize text from the scanned pages",
        "use EasyOCR to read this PDF",
        "اعمل مسح ضوئي لملف PDF واستخرج النص",
        "استخرج النص من صفحات الكتاب المصورة",
    ],
    "search_pdf_knowledge": [
        "search the learned PDF information",
        "what does the PDF say about this topic",
        "ابحث في معلومات الكتب",
        "ماذا يقول الملف عن هذا الموضوع",
        "دور في الكتب عن المعلومة دي",
    ],
    "answer_pdf_question": [
        "answer a question using the learned book",
        "what does this book say about this subject",
        "ask the PDF a question and answer from its text",
        "جاوب على سؤالي من الكتاب",
        "ما إجابة هذا السؤال في الكتاب",
        "اشرح المعلومة اعتمادا على محتوى الكتاب",
    ],
}

QA_MODEL = "mrm8488/bert-multi-cased-finetuned-xquadv1"


TOOL_SCHEMAS = [
    {
        "name": "find_and_copy_pdf",
        "description": (
            "Search the current user's profile by PDF filename, or use a provided "
            "PDF path, copy one file into the library books folder, extract a "
            "valid ISBN from its selectable text, and register it in the catalog. "
            "If title or author metadata is missing, provide those details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {"type": "string"},
                "pdf_path": {"type": "string"},
                "title": {"type": "string"},
                "author": {"type": "string"},
            },
        },
    },
    {
        "name": "add_book",
        "description": "Add a book and copy its PDF into the library.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                "pdf_path": {"type": "string"},
            },
            "required": ["title", "author", "pdf_path"],
        },
    },
    {
        "name": "remove_book",
        "description": "Remove a book by ISBN.",
        "parameters": {
            "type": "object",
            "properties": {"isbn": {"type": "string"}},
            "required": ["isbn"],
        },
    },
    {
        "name": "modify_book",
        "description": "Change a book's title and author by ISBN.",
        "parameters": {
            "type": "object",
            "properties": {
                "isbn": {"type": "string"},
                "new_title": {"type": "string"},
                "new_author": {"type": "string"},
            },
            "required": ["isbn", "new_title", "new_author"],
        },
    },
    {
        "name": "search_book",
        "description": "Search books by title or author.",
        "parameters": {
            "type": "object",
            "properties": {"search_term": {"type": "string"}},
            "required": ["search_term"],
        },
    },
    {"name": "see_all_books", "description": "List all books.", "parameters": {"type": "object", "properties": {}}},
    {
        "name": "add_member",
        "description": "Register a library member.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "remove_member",
        "description": "Remove a member by member ID.",
        "parameters": {
            "type": "object",
            "properties": {"member_id": {"type": "integer"}},
            "required": ["member_id"],
        },
    },
    {"name": "see_all_members", "description": "List all library members.", "parameters": {"type": "object", "properties": {}}},
    {
        "name": "loan_book",
        "description": "Loan a book to a member.",
        "parameters": {
            "type": "object",
            "properties": {
                "member_id": {"type": "integer"},
                "isbn": {"type": "string"},
            },
            "required": ["member_id", "isbn"],
        },
    },
    {"name": "see_all_loans", "description": "List all active loans.", "parameters": {"type": "object", "properties": {}}},
    {
        "name": "return_book",
        "description": "Return a book for a member.",
        "parameters": {
            "type": "object",
            "properties": {
                "member_id": {"type": "integer"},
                "isbn": {"type": "string"},
            },
            "required": ["member_id", "isbn"],
        },
    },
    {
        "name": "read_pdf",
        "description": "Read and extract text from a PDF path or a library book ISBN.",
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string"},
                "isbn": {"type": "string"},
            },
        },
    },
    {
        "name": "analyze_pdf",
        "description": "Extract PDF metadata, ISBNs, page count, and a text excerpt.",
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string"},
                "isbn": {"type": "string"},
            },
        },
    },
    {
        "name": "learn_pdf",
        "description": "Extract a PDF and save its text to the local searchable knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string"},
                "isbn": {"type": "string"},
            },
        },
    },
    {
        "name": "ocr_pdf",
        "description": (
            "Run EasyOCR on scanned PDF pages and save the recognized text for search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string"},
                "isbn": {"type": "string"},
            },
        },
    },
    {
        "name": "search_pdf_knowledge",
        "description": "Search PDF text previously saved in the local knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "answer_pdf_question",
        "description": "Answer a question using learned PDF text and cite its source page.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
]


class PDFKnowledgeBase:
    def __init__(self, library, data_file=None):
        self.library = library
        self.data_file = (
            Path(data_file)
            if data_file
            else Path(__file__).with_name("pdf_knowledge.json")
        )
        self._qa_pipeline = None
        self._ocr_readers = {}

    def _resolve_pdf(self, pdf_path=None, isbn=None):
        if isbn:
            book = next(
                (book for book in self.library.books if book.isbn == str(isbn)),
                None,
            )
            if book is None:
                raise ValueError(f"No book found with ISBN '{isbn}'.")
            if not book.pdf_filename:
                raise ValueError(f"No PDF is saved for ISBN '{isbn}'.")
            path = self.library.get_pdf_path(book.pdf_filename)
        elif pdf_path:
            path = Path(pdf_path).expanduser()
        else:
            raise ValueError("Provide either pdf_path or isbn.")

        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files can be read.")
        return path.resolve()

    def extract(self, pdf_path=None, isbn=None):
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        path = self._resolve_pdf(pdf_path, isbn)
        try:
            reader = PdfReader(str(path))
            pages = [
                {"page": number, "text": page.extract_text() or ""}
                for number, page in enumerate(reader.pages, start=1)
            ]
            text = "\n".join(page["text"] for page in pages).strip()
        except PdfReadError as error:
            raise ValueError(f"Could not read the PDF file: {error}") from error
        metadata = reader.metadata or {}
        return {
            "path": str(path),
            "page_count": len(reader.pages),
            "metadata": {
                "title": metadata.get("/Title"),
                "author": metadata.get("/Author"),
                "subject": metadata.get("/Subject"),
            },
            "pages": pages,
            "text": text,
        }

    def read_pdf(self, pdf_path=None, isbn=None, max_chars=20000):
        result = self.extract(pdf_path, isbn)
        text = result["text"]
        result["text"] = text[:max_chars]
        result["truncated"] = len(text) > max_chars
        return result

    def analyze_pdf(self, pdf_path=None, isbn=None):
        result = self.extract(pdf_path, isbn)
        isbn_matches = re.findall(
            r"(?<!\d)(?:97[89][ -]?)?(?:\d[ -]?){9,12}\d(?!\d)",
            result["text"],
        )
        normalized_isbns = list(
            dict.fromkeys(re.sub(r"\D", "", value) for value in isbn_matches)
        )
        return {
            "path": result["path"],
            "page_count": result["page_count"],
            "metadata": result["metadata"],
            "isbn_candidates": normalized_isbns,
            "text_characters": len(result["text"]),
            "excerpt": result["text"][:1500],
        }

    def learn_pdf(self, pdf_path=None, isbn=None):
        extracted = self.extract(pdf_path, isbn)
        if not extracted["text"]:
            raise ValueError(
                "No selectable text was found in the PDF. Use the ocr_pdf tool "
                "to recognize scanned pages."
            )
        return self._save_extracted_document(extracted)

    def ocr_pdf(self, pdf_path=None, isbn=None):
        extracted = self.extract(pdf_path, isbn)
        pages_to_ocr = [
            page for page in extracted["pages"] if not page["text"].strip()
        ]
        if pages_to_ocr:
            pages_by_number = {page["page"]: page for page in pages_to_ocr}
            rendered_pages = self._render_pdf_pages(
                extracted["path"],
                [page["page"] for page in pages_to_ocr],
            )
            try:
                for page_number, image in rendered_pages:
                    try:
                        pages_by_number[page_number]["text"] = (
                            self._recognize_image(image).strip()
                        )
                    finally:
                        image.close()
            finally:
                rendered_pages.close()

        extracted["text"] = "\n".join(
            page["text"] for page in extracted["pages"] if page["text"].strip()
        ).strip()
        if not extracted["text"]:
            raise ValueError("OCR did not recognize any text in the PDF.")

        result = self._save_extracted_document(extracted)
        return {
            **result,
            "engine": "easyocr",
            "ocr_pages": [page["page"] for page in pages_to_ocr],
        }

    @staticmethod
    def _render_pdf_pages(pdf_path, page_numbers):
        try:
            import pymupdf
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "PDF OCR rendering requires PyMuPDF and Pillow. Install the OCR "
                "requirements with: python -m pip install -r requirements-ocr.txt"
            ) from error

        document = pymupdf.open(str(pdf_path))
        try:
            for page_number in page_numbers:
                page = document.load_page(page_number - 1)
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                yield page_number, image
        finally:
            document.close()

    def _recognize_image(self, image):
        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError(
                "EasyOCR is not installed. Install it with: "
                "python -m pip install -r requirements-ocr.txt"
            ) from error
        reader = self._ocr_readers.get("easyocr")
        if reader is None:
            reader = easyocr.Reader(["en", "ar"], gpu=False)
            self._ocr_readers["easyocr"] = reader
        recognized_lines = reader.readtext(
            np.asarray(image),
            detail=0,
            paragraph=True,
        )
        return "\n".join(str(line) for line in recognized_lines)

    def _save_extracted_document(self, extracted):
        data = {"documents": []}
        if self.data_file.exists():
            with self.data_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
                raise ValueError("The PDF knowledge file must contain a 'documents' list.")

        source = extracted["path"]
        documents = [
            document
            for document in data["documents"]
            if isinstance(document, dict) and document.get("source") != source
        ]
        documents.append(
            {
                "source": source,
                "page_count": extracted["page_count"],
                "metadata": extracted["metadata"],
                "pages": extracted.get("pages", []),
                "text": extracted["text"],
            }
        )
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = self.data_file.with_suffix(".tmp")
        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump({"documents": documents}, file, ensure_ascii=False, indent=2)
        temporary_file.replace(self.data_file)
        return {
            "source": source,
            "page_count": extracted["page_count"],
            "characters_indexed": len(extracted["text"]),
            "knowledge_file": str(self.data_file),
        }

    def learn_library_pdfs(self, books_folder=None):
        folders = [
            Path(books_folder or self.library.books_folder).expanduser().resolve()
        ]
        if books_folder is None:
            loaned_folder = getattr(self.library, "loaned_books_folder", None)
            if loaned_folder is not None:
                resolved_loaned_folder = Path(loaned_folder).expanduser().resolve()
                if resolved_loaned_folder not in folders:
                    folders.append(resolved_loaned_folder)

        report = {
            "scanned": 0,
            "indexed": [],
            "unchanged": [],
            "removed": [],
            "errors": [],
            "topics": [],
        }
        for folder in folders:
            if folder.exists() and not folder.is_dir():
                raise NotADirectoryError(
                    f"Library books path is not a directory: {folder}"
                )

        state_file = self.data_file.with_name("pdf_library_index.json")
        previous_files = {}
        previous_topics = []
        if state_file.exists():
            with state_file.open("r", encoding="utf-8") as file:
                state = json.load(file)
            if (
                not isinstance(state, dict)
                or not isinstance(state.get("files"), dict)
                or not isinstance(state.get("topics", []), list)
            ):
                raise ValueError("The PDF library index file has an invalid format.")
            previous_files = state["files"]
            previous_topics = state.get("topics", [])

        pdf_paths = sorted(
            (path for folder in folders if folder.is_dir() for path in folder.glob("*.pdf")),
            key=lambda path: path.name.casefold(),
        )
        report["scanned"] = len(pdf_paths)
        present_sources = {str(path.resolve()) for path in pdf_paths}
        current_files = {}

        for path in pdf_paths:
            source = str(path.resolve())
            try:
                stat = path.stat()
            except OSError as error:
                report["errors"].append({"source": source, "error": str(error)})
                if source in previous_files:
                    current_files[source] = previous_files[source]
                continue

            signature = {"size": stat.st_size, "modified_ns": stat.st_mtime_ns}
            previous_signature = previous_files.get(source)
            if previous_signature == signature:
                current_files[source] = signature
                report["unchanged"].append(source)
                continue

            try:
                self.learn_pdf(pdf_path=source)
            except (OSError, ValueError) as error:
                report["errors"].append({"source": source, "error": str(error)})
                if previous_signature is not None:
                    current_files[source] = previous_signature
                continue

            current_files[source] = signature
            report["indexed"].append(source)

        removed_sources = set(previous_files) - present_sources
        if removed_sources:
            try:
                self._forget_sources(removed_sources)
            except (OSError, ValueError) as error:
                report["errors"].append(
                    {"source": None, "error": f"Could not remove deleted PDFs from the index: {error}"}
                )
                for source in removed_sources:
                    current_files[source] = previous_files[source]
            else:
                report["removed"] = sorted(removed_sources)

        topics = previous_topics
        if report["indexed"] or report["removed"]:
            try:
                topics = self._discover_library_topics(current_files)
            except ValueError as error:
                report["errors"].append(
                    {"source": None, "error": f"Could not cluster PDF topics: {error}"}
                )
                topics = []
        report["topics"] = topics
        self._write_json_atomic(
            state_file,
            {"files": current_files, "topics": topics},
        )
        return report

    def _forget_sources(self, sources):
        if not self.data_file.exists():
            return
        with self.data_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
            raise ValueError("The PDF knowledge file must contain a 'documents' list.")
        data["documents"] = [
            document
            for document in data["documents"]
            if not isinstance(document, dict) or document.get("source") not in sources
        ]
        self._write_json_atomic(self.data_file, data)

    def _discover_library_topics(self, indexed_files, max_topics=8):
        if not self.data_file.exists():
            return []
        with self.data_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
            raise ValueError("The PDF knowledge file must contain a 'documents' list.")

        sources = set(indexed_files)
        chunks = [
            chunk
            for document in data["documents"]
            if isinstance(document, dict) and document.get("source") in sources
            for chunk in self._document_chunks(document)
        ]
        if len(chunks) < 2:
            return []

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            stop_words="english",
        )
        try:
            matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
        except ValueError as error:
            raise ValueError(f"Could not build a vocabulary from book text: {error}") from error

        cluster_count = min(
            max_topics,
            len(chunks),
            max(2, int(len(chunks) ** 0.5)),
        )
        labels = cluster_sparse_vectors(matrix, cluster_count)
        features = vectorizer.get_feature_names_out()
        topics = []
        for cluster in range(cluster_count):
            indices = [
                index for index, label in enumerate(labels) if int(label) == cluster
            ]
            if not indices:
                continue
            term_weights = {}
            for index in indices:
                row = matrix.getrow(index)
                for feature, weight in zip(row.indices, row.data):
                    feature_index = int(feature)
                    term_weights[feature_index] = (
                        term_weights.get(feature_index, 0.0) + float(weight)
                    )
            keywords = sorted(
                term_weights,
                key=term_weights.get,
                reverse=True,
            )[:6]
            topics.append(
                {
                    "cluster": cluster,
                    "keywords": [str(features[index]) for index in keywords],
                    "chunk_count": len(indices),
                    "books": sorted(
                        {
                            Path(chunks[index]["source"]).name
                            for index in indices
                            if chunks[index]["source"]
                        }
                    ),
                }
            )
        return topics

    @staticmethod
    def _write_json_atomic(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = path.with_suffix(".tmp")
        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        temporary_file.replace(path)

    @staticmethod
    def _document_chunks(document):
        chunks = []
        pages = document.get("pages")
        if isinstance(pages, list) and pages:
            page_texts = (
                (page.get("page", number), page.get("text"))
                for number, page in enumerate(pages, start=1)
                if isinstance(page, dict)
            )
        elif isinstance(document.get("text"), str):
            page_texts = [(None, document["text"])]
        else:
            return chunks

        for page_number, page_text in page_texts:
            if not isinstance(page_text, str):
                continue
            for offset in range(0, len(page_text), 900):
                text = page_text[offset:offset + 1100]
                if text.strip():
                    chunks.append(
                        {
                            "source": document.get("source"),
                            "page": page_number,
                            "text": text,
                        }
                    )
        return chunks

    def _rank_chunks(self, query, limit):
        if not self.data_file.exists():
            return []
        with self.data_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
            raise ValueError("The PDF knowledge file must contain a 'documents' list.")

        chunks = [
            chunk
            for document in data["documents"]
            if isinstance(document, dict)
            for chunk in self._document_chunks(document)
        ]
        if not chunks:
            return []

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=True,
        )
        matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
        query_vector = vectorizer.transform([query])
        scores = (matrix @ query_vector.T).toarray().ravel()
        ranked = sorted(
            (
                (float(score), chunk)
                for score, chunk in zip(scores, chunks)
                if score > 0
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "score": score,
                "text": chunk["text"],
            }
            for score, chunk in ranked[:limit]
        ]

    def search(self, query, limit=3):
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty.")
        return self._rank_chunks(query, limit)

    def _get_qa_pipeline(self):
        if self._qa_pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as error:
                raise RuntimeError(
                    "PDF question answering requires transformers and torch. "
                    "Install the project requirements and try again."
                ) from error
            self._qa_pipeline = pipeline(
                "question-answering",
                model=QA_MODEL,
            )
        return self._qa_pipeline

    def answer_pdf_question(self, question, limit=5):
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        candidates = self._rank_chunks(question, limit)
        if not candidates:
            return {
                "found": False,
                "answer": "لا توجد ملفات PDF متعلمة. استخدم learn_pdf أولاً.",
                "sources": [],
            }

        qa_pipeline = self._get_qa_pipeline()
        best_answer = None
        for candidate in candidates:
            prediction = qa_pipeline(
                question=question,
                context=candidate["text"],
            )
            answer = prediction["answer"].strip()
            confidence = float(prediction["score"])
            if answer and (
                best_answer is None or confidence > best_answer["confidence"]
            ):
                best_answer = {
                    "answer": answer,
                    "confidence": confidence,
                    "source": candidate["source"],
                    "page": candidate["page"],
                    "evidence": candidate["text"],
                }

        if best_answer is None or best_answer["confidence"] < 0.1:
            return {
                "found": False,
                "answer": "لم أجد إجابة موثوقة لهذا السؤال في النص المسترجع من الكتب.",
                "sources": [],
            }

        return {
            "found": True,
            "answer": best_answer["answer"],
            "confidence": best_answer["confidence"],
            "sources": [
                {
                    "source": best_answer["source"],
                    "page": best_answer["page"],
                    "evidence": best_answer["evidence"],
                }
            ],
        }


class LocalLibraryAgent:
    def __init__(self, library, knowledge_base=None, learning_memory=None):
        self.library = library
        self.knowledge_base = knowledge_base or PDFKnowledgeBase(library)
        self.text_generator = LocalTextGenerator()
        self.learning_memory = learning_memory or SelfLearningMemory()
        self.last_generated_answer = None
        self.response_language = "ar"
        self._language_preference_explicit = False
        examples = [
            (phrase, intent)
            for intent, phrases in INTENT_EXAMPLES.items()
            for phrase in phrases
        ]
        examples.extend(
            (phrase, "general_chat")
            for phrase in self.text_generator.intent_examples
        )
        valid_tools = {tool["name"] for tool in TOOL_SCHEMAS}
        examples.extend(
            (prompt, tool)
            for prompt, tool in self.learning_memory.training_examples()
            if tool in valid_tools
        )
        examples.extend(
            (item["question"], "general_chat")
            for item in self.learning_memory.verified_answers
        )
        self.intent_model = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True),
            LogisticRegression(max_iter=1000),
        )
        self.intent_model.fit(
            [phrase for phrase, _ in examples],
            [intent for _, intent in examples],
        )
        self.tool_schemas = {tool["name"]: tool for tool in TOOL_SCHEMAS}
        self.tools = {
            "find_and_copy_pdf": library.find_and_copy_pdf,
            "add_book": library.add_book,
            "remove_book": library.remove_book,
            "modify_book": library.modify_book,
            "search_book": library.search_book,
            "see_all_books": library.see_all_books,
            "add_member": library.add_member,
            "remove_member": library.remove_member,
            "see_all_members": library.see_all_members,
            "loan_book": library.loan_book,
            "see_all_loans": library.see_all_loans,
            "return_book": library.return_book,
            "read_pdf": self.knowledge_base.read_pdf,
            "analyze_pdf": self.knowledge_base.analyze_pdf,
            "learn_pdf": self.knowledge_base.learn_pdf,
            "ocr_pdf": self.knowledge_base.ocr_pdf,
            "search_pdf_knowledge": self.knowledge_base.search,
            "answer_pdf_question": self.knowledge_base.answer_pdf_question,
        }

    def call_tool(self, name, arguments):
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object.")
        schema = self.tool_schemas[name]["parameters"]
        properties = schema.get("properties", {})
        unknown = set(arguments) - set(properties)
        if unknown:
            raise ValueError(f"Unexpected arguments for {name}: {', '.join(sorted(unknown))}")
        for argument, value in arguments.items():
            expected_type = properties[argument]["type"]
            if expected_type == "string" and not isinstance(value, str):
                raise ValueError(f"Argument '{argument}' must be text.")
            if expected_type == "integer" and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"Argument '{argument}' must be an integer.")
        missing = [
            argument
            for argument in schema.get("required", [])
            if argument not in arguments
        ]
        if missing:
            raise ValueError(f"Missing arguments for {name}: {', '.join(missing)}")

        output = StringIO()
        with redirect_stdout(output):
            result = self.tools[name](**arguments)
        return {"tool": name, "result": result, "output": output.getvalue().strip()}

    @staticmethod
    def _isbn(text):
        match = re.search(r"(?<!\d)\d{9,13}(?!\d)", text)
        return match.group(0) if match else None

    @staticmethod
    def _member_id(text):
        match = re.search(
            r"(?:member(?:\s+id)?|عضو|العضو|رقم\s+العضو)\s*[:#]?\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        return int(match.group(1)) if match else None

    @staticmethod
    def _labeled_value(text, labels):
        for label in labels:
            match = re.search(
                rf"(?:{label})\s*[:=]\s*(?:['\"]([^'\"]+)['\"]|(.+?))"
                rf"(?=\s+(?:title|author|new title|new author|pdf|pdf_path|"
                rf"عنوان|مؤلف|ملف|مسار)\s*[:=]|$)",
                text,
                re.IGNORECASE,
            )
            if match:
                return (match.group(1) or match.group(2)).strip().strip("\"'")
        return None

    @classmethod
    def _pdf_path(cls, text):
        match = re.search(r"""['"]([^'"]+\.pdf)['"]""", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(
            r"(?:(?:[A-Za-z]:\\|\.{1,2}[\\/]|/)[^\r\n]*?\.pdf)",
            text,
            re.IGNORECASE,
        )
        return match.group(0).strip().rstrip(".,;") if match else None

    @classmethod
    def _arguments_for(cls, tool_name, prompt):
        arguments = {}
        isbn = cls._isbn(prompt)
        member_id = cls._member_id(prompt)
        pdf_path = cls._pdf_path(prompt)

        if tool_name in {"remove_book", "modify_book", "loan_book", "return_book"} and isbn:
            arguments["isbn"] = isbn
        if tool_name in {"remove_member", "loan_book", "return_book"} and member_id:
            arguments["member_id"] = member_id
        if tool_name in {"read_pdf", "analyze_pdf", "learn_pdf", "ocr_pdf"}:
            if isbn:
                arguments["isbn"] = isbn
            elif pdf_path:
                arguments["pdf_path"] = pdf_path
        if tool_name == "search_book":
            value = cls._labeled_value(prompt, [r"search(?:\s+term)?", r"ابحث\s+عن", r"دور\s+على"])
            if value is None:
                match = re.search(
                    r"(?:search|find|look for|ابحث عن|دور على)"
                    r"(?:\s+(?:a\s+)?book|\s+كتاب)?\s+(.+)$",
                    prompt,
                    re.IGNORECASE,
                )
                value = match.group(1).strip(" .؟?\"'") if match else None
            if value:
                arguments["search_term"] = value
        if tool_name == "find_and_copy_pdf":
            if pdf_path:
                arguments["pdf_path"] = pdf_path
            else:
                value = cls._labeled_value(
                    prompt,
                    [r"search term", r"filename", r"file name", r"اسم الملف", r"اسم"],
                )
                if value is None:
                    value = prompt
                    english_prefixes = (
                        r"^\s*(?:please\s+)?"
                        r"(?:find|locate|search|copy|add|import|look\s+for)\s+"
                        r"(?:(?:in|through)\s+)?(?:(?:my|all)\s+)?folders?\s+for\s+",
                        r"^\s*(?:please\s+)?"
                        r"(?:find|locate|search(?:\s+for)?|copy|add|import|look\s+for)\s+",
                    )
                    arabic_prefix = (
                        r"^\s*(?:من فضلك\s*)?"
                        r"(?:ابحث|دور|هات|انسخ|ضيف|اضف|أضف|استورد)\s*"
                        r"(?:(?:في|داخل)\s+(?:كل\s+)?(?:مجلداتي|المجلدات|المجلد)\s*)?"
                        r"(?:عن|على)?\s*"
                    )
                    for prefix in english_prefixes:
                        value, count = re.subn(
                            prefix,
                            "",
                            value,
                            count=1,
                            flags=re.IGNORECASE,
                        )
                        if count:
                            break
                    else:
                        value = re.sub(arabic_prefix, "", value, count=1)
                    value = re.sub(
                        r"^(?:(?:the|a|an)\s+)?(?:pdf(?:\s+file)?|file)\s*"
                        r"(?:(?:named|called|matching)\s+)?",
                        "",
                        value,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                    value = re.sub(
                        r"^(?:الملف|ملف(?:\s+pdf)?|pdf)\s*"
                        r"(?:(?:باسم|اسمه|اسم)\s*)?",
                        "",
                        value,
                        flags=re.IGNORECASE,
                    )
                    value = re.sub(
                        r"\s+(?:and\s+(?:copy|add|put)\b.*|to\s+(?:the\s+)?library.*|"
                        r"into\s+(?:the\s+)?library.*|وانسخ(?:ه|ها)?.*|"
                        r"وانقل(?:ه|ها)?.*|للمكتبة.*|إلى\s+المكتبة.*|الى\s+المكتبة.*)$",
                        "",
                        value,
                        flags=re.IGNORECASE,
                    )
                    value = value.strip().strip("\"'")
                if value:
                    arguments["search_term"] = value
            for key, labels in {
                "title": [r"title", r"عنوان(?:\s+الكتاب)?"],
                "author": [r"author", r"مؤلف(?:\s+الكتاب)?"],
            }.items():
                value = cls._labeled_value(prompt, labels)
                if value:
                    arguments[key] = value
        if tool_name == "search_pdf_knowledge":
            value = cls._labeled_value(prompt, [r"query", r"search", r"ابحث\s+عن", r"دور\s+على"])
            if value is None:
                match = re.search(r"(?:about|عن|حول)\s+(.+)$", prompt, re.IGNORECASE)
                value = match.group(1).strip(" .؟?\"'") if match else None
            if value:
                arguments["query"] = value
        if tool_name == "answer_pdf_question":
            value = cls._labeled_value(prompt, [r"question", r"سؤال"])
            if value is None:
                value = re.sub(
                    r"^\s*(?:please\s+)?(?:answer|ask the book|ask the PDF)"
                    r"(?:\s+this question)?\s*[:,-]?\s*",
                    "",
                    prompt,
                    flags=re.IGNORECASE,
                )
                value = re.sub(
                    r"^\s*(?:جاوب(?:ني)?(?:\s+على)?\s+(?:سؤالي|السؤال)"
                    r"|اسأل الكتاب|اسأل الملف)"
                    r"(?:\s+(?:من|في)\s+(?:الكتاب|الملف))?\s*[:،,-]?\s*",
                    "",
                    value,
                )
            if value.strip():
                arguments["question"] = value.strip()
        if tool_name == "add_book":
            for key, labels in {
                "title": [r"title", r"عنوان(?:\s+الكتاب)?"],
                "author": [r"author", r"مؤلف(?:\s+الكتاب)?"],
            }.items():
                value = cls._labeled_value(prompt, labels)
                if value:
                    arguments[key] = value
            if pdf_path:
                arguments["pdf_path"] = pdf_path
        if tool_name == "modify_book":
            for key, labels in {
                "new_title": [r"new title", r"title", r"العنوان الجديد", r"عنوان جديد"],
                "new_author": [r"new author", r"author", r"المؤلف الجديد", r"مؤلف جديد"],
            }.items():
                value = cls._labeled_value(prompt, labels)
                if value:
                    arguments[key] = value
        if tool_name == "add_member":
            value = cls._labeled_value(prompt, [r"name", r"اسم العضو", r"اسم"])
            if value:
                arguments["name"] = value
        return arguments

    def infer_tool(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a non-empty request.")
        pdf_path = self._pdf_path(prompt)
        if re.search(
            r"\b(?:run|perform|do|use)\s+ocr\b|"
            r"\bocr\s+(?:this|the|on|for|pdf)\b|"
            r"\b(?:use|using)\s+easyocr\b|"
            r"\bscan\b.{0,30}\bpdf\b|"
            r"\brecognize text\b.{0,30}\b(?:pdf|scanned|pages)\b|"
            r"optical character recognition|"
            r"(?:مسح\s+ضوئي|امسح\s+ضوئي|اعمل\s+مسح|التعرف\s+الضوئي)",
            prompt,
            re.IGNORECASE,
        ):
            return {
                "tool": "ocr_pdf",
                "confidence": 1.0,
                "arguments": self._arguments_for("ocr_pdf", prompt),
            }
        has_catalog_details = bool(
            self._labeled_value(prompt, [r"title", r"عنوان(?:\s+الكتاب)?"])
            or self._labeled_value(prompt, [r"author", r"مؤلف(?:\s+الكتاب)?"])
        )
        if not has_catalog_details and (
            (
                pdf_path
                and re.search(
                    r"\b(?:add|copy|import)\b|(?:ضيف|اضف|أضف|انسخ|استورد)",
                    prompt,
                    re.IGNORECASE,
                )
            )
            or re.search(
                r"\b(?:add|copy|import)\b.*(?:\.pdf\b|\bpdf\b)|"
                r"(?:ضيف|اضف|أضف|انسخ|استورد).*(?:\.pdf|pdf|ملف)",
                prompt,
                re.IGNORECASE,
            )
        ):
            return {
                "tool": "find_and_copy_pdf",
                "confidence": 1.0,
                "arguments": self._arguments_for("find_and_copy_pdf", prompt),
            }
        if pdf_path and re.search(
            r"\b(?:read|learn|open|load|index|summari[sz]e)\b|"
            r"(?:اقرأ|اقرا|قراءة|تعلم|احفظ|افتح|لخص|لخّص)",
            prompt,
            re.IGNORECASE,
        ):
            tool_name = (
                "analyze_pdf"
                if re.search(
                    r"\b(?:analy[sz]e|inspect)\b|(?:حلل|حلّل)",
                    prompt,
                    re.IGNORECASE,
                )
                else "learn_pdf"
            )
            return {
                "tool": tool_name,
                "confidence": 1.0,
                "arguments": {"pdf_path": pdf_path},
            }
        if re.search(
            r"\b(?:find|search|locate)\b.*\b(?:pdf|file)\b.*"
            r"\b(?:folders?|filename|file name|named|called|copy|library)\b|"
            r"\b(?:find|search|locate)\b.*\.pdf\b|"
            r"\b(?:find|search)\s+(?:my\s+)?folders?\b.*\b(?:pdf|file)\b|"
            r"(?:ابحث|دور).*(?:pdf|ملف).*(?:\.pdf|مجلد|اسم|انسخ|المكتبة)|"
            r"(?:ابحث|دور).*\S+\.pdf\b|"
            r"(?:انسخ|هات).*(?:pdf|ملف).*(?:مجلد|المكتبة)",
            prompt,
            re.IGNORECASE,
        ):
            return {
                "tool": "find_and_copy_pdf",
                "confidence": 1.0,
                "arguments": self._arguments_for("find_and_copy_pdf", prompt),
            }
        if self.text_generator.matches_dataset_prompt(prompt):
            return {
                "tool": "general_chat",
                "confidence": 1.0,
                "arguments": {},
            }
        probabilities = self.intent_model.predict_proba([prompt])[0]
        best_index = probabilities.argmax()
        confidence = float(probabilities[best_index])
        tool_name = self.intent_model.classes_[best_index]
        if confidence < 0.08:
            tool_name = "general_chat"
        if (
            tool_name in {"answer_pdf_question", "search_pdf_knowledge"}
            and not self._mentions_book_or_pdf(prompt)
        ):
            tool_name = "general_chat"
        return {
            "tool": tool_name,
            "confidence": confidence,
            "arguments": (
                {}
                if tool_name == "general_chat"
                else self._arguments_for(tool_name, prompt)
            ),
        }

    @staticmethod
    def _mentions_book_or_pdf(prompt):
        return bool(
            re.search(
                r"\b(?:book|pdf|document|chapter|passage)\b|"
                r"(?:الكتاب|كتاب|الملف|ملف|الفصل|النص)",
                prompt,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _requested_language(prompt):
        if re.search(
            r"\b(?:speak|talk|respond|reply|use)\b.*\benglish\b",
            prompt,
            re.IGNORECASE,
        ):
            return "en"
        if re.search(
            r"(?:تكلم|تحدث|رد|استخدم).*(?:العربية|عربي|الإنجليزية|انجليزي)|"
            r"(?:بالعربية|بالعربي|بالإنجليزية|بالانجليزي)",
            prompt,
        ):
            return "ar" if re.search(r"عربي|العربية", prompt) else "en"
        return None

    @staticmethod
    def _is_farewell(prompt):
        normalized = prompt.casefold().strip(" \t\r\n.,!?؟")
        if re.search(r"\b(?:bye|goodbye|farewell)\b|\bsee\s+(?:you|ya)\b", normalized):
            return True
        if normalized in {"no thanks", "no thank you", "that's all", "that is all"}:
            return True
        return bool(
            re.search(
                r"(?:مع السلامة|إلى اللقاء|الى اللقاء|وداعا|وداعًا|خلاص شكرا|خلاص شكرًا)",
                normalized,
            )
        )

    @staticmethod
    def _is_remember_answer_request(prompt):
        return bool(
            re.search(
                r"\b(?:remember|save)\s+(?:(?:this|that|the previous|last)\s+)?"
                r"(?:answer|response)\b|"
                r"(?:احفظ|سجل|علّم|علم)\s+(?:الإجابة|الاجابة|الرد)",
                prompt,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _greeting_language(prompt):
        greeting = prompt.casefold().strip(" \t\r\n.,!?؟")
        if greeting in {
            "hello",
            "hi",
            "hey",
            "hello there",
            "hi there",
            "good morning",
            "good afternoon",
            "good evening",
        }:
            return "en"
        if greeting in {
            "اهلا",
            "أهلا",
            "مرحبا",
            "السلام عليكم",
            "صباح الخير",
            "مساء الخير",
        }:
            return "ar"
        return None

    @staticmethod
    def _prompt_language(prompt):
        return "ar" if re.search(r"[\u0600-\u06ff]", prompt) else "en"

    def handle_prompt(self, prompt, input_fn=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a non-empty request.")

        requested_language = self._requested_language(prompt)
        if requested_language:
            self.response_language = requested_language
            self._language_preference_explicit = True
            message = (
                "Sure, I'll speak English. I can help with books, members, loans, "
                "and questions about learned PDFs."
                if requested_language == "en"
                else "حاضر، سأتحدث بالعربية. أستطيع مساعدتك في الكتب والأعضاء "
                "والإعارات والأسئلة عن ملفات PDF التي تعلمتها."
            )
            return {"tool": None, "message": message}

        if self._is_remember_answer_request(prompt):
            if self.last_generated_answer is None:
                message = (
                    "There is no recent generated answer to remember."
                    if self.response_language == "en"
                    else "لا توجد إجابة مولّدة حديثة لحفظها."
                )
                return {"tool": None, "message": message}
            self.learning_memory.record_verified_answer(
                self.last_generated_answer["question"],
                self.last_generated_answer["answer"],
            )
            self.last_generated_answer = None
            message = (
                "Saved. I will use this answer as an example next time."
                if self.response_language == "en"
                else "تم الحفظ بعد تأكيدك، وسأستخدم الإجابة مثالًا عند الحاجة."
            )
            return {"tool": None, "message": message}

        if self._is_farewell(prompt):
            if not self._language_preference_explicit:
                self.response_language = self._prompt_language(prompt)
            message = (
                "Goodbye! The assistant session is closed."
                if self.response_language == "en"
                else "مع السلامة! تم إنهاء جلسة المساعد."
            )
            return {"tool": None, "message": message, "end_agent": True}

        verified_answer = self.learning_memory.find_verified_answer(prompt)
        if verified_answer is not None:
            return {"tool": "verified_memory", "message": verified_answer}

        greeting_language = self._greeting_language(prompt)
        if greeting_language and not self._language_preference_explicit:
            self.response_language = greeting_language

        inferred = self.infer_tool(prompt)
        tool_name = inferred["tool"]
        if tool_name == "general_chat":
            if not self._language_preference_explicit:
                self.response_language = self._prompt_language(prompt)
            answer = self.text_generator.generate(
                prompt,
                language=self.response_language,
                verified_examples=[
                    {
                        "input": item["question"],
                        "response": item["answer"],
                    }
                    for item in self.learning_memory.verified_answers
                ],
            )
            self.last_generated_answer = {
                "question": prompt.strip(),
                "answer": answer,
            }
            return {
                "tool": "general_chat",
                "message": answer,
            }
        if tool_name is None:
            return inferred
        schema = self.tool_schemas[tool_name]["parameters"]
        arguments = inferred["arguments"]
        missing = [
            name for name in schema.get("required", []) if name not in arguments
        ]
        if missing and input_fn is None:
            return {
                **inferred,
                "missing_arguments": missing,
                "message": (
                    f"I need these details to continue: {', '.join(missing)}"
                    if self.response_language == "en"
                    else f"أحتاج هذه البيانات لاستدعاء الأداة: {', '.join(missing)}"
                ),
            }
        for name in missing:
            arguments[name] = input_fn(f"{name}: ").strip()
        if (
            tool_name == "find_and_copy_pdf"
            and not any(name in arguments for name in ("search_term", "pdf_path"))
        ):
            if input_fn is None:
                return {
                    **inferred,
                    "missing_arguments": ["search_term or pdf_path"],
                    "message": (
                        "Provide a PDF filename or a full path inside your user profile."
                        if self.response_language == "en"
                        else "أرسل اسم ملف PDF أو مساره الكامل داخل مجلد حسابك."
                    ),
                }
            value = input_fn("PDF filename or full path: ").strip()
            if not value:
                raise ValueError("Provide a PDF filename or full path.")
            path = self._pdf_path(value)
            arguments.update(
                {"pdf_path": path}
                if path
                else {"search_term": value}
            )
        if tool_name in {"read_pdf", "analyze_pdf", "learn_pdf", "ocr_pdf"} and not arguments:
            if input_fn is None:
                return {
                    **inferred,
                    "missing_arguments": ["pdf_path or isbn"],
                    "message": (
                        "Provide a PDF path or the ISBN of a saved book."
                        if self.response_language == "en"
                        else "أرسل مسار PDF أو ISBN لكتاب محفوظ."
                    ),
                }
            arguments["pdf_path"] = input_fn("pdf_path (or leave blank for ISBN): ").strip()
            if not arguments["pdf_path"]:
                arguments["isbn"] = input_fn("isbn: ").strip()
                arguments.pop("pdf_path")
        result = self.call_tool(tool_name, arguments)
        copy_result = result.get("result")
        if (
            tool_name == "find_and_copy_pdf"
            and isinstance(copy_result, dict)
            and copy_result.get("needs_book_details")
        ):
            missing_details = copy_result["needs_book_details"]
            if input_fn is None:
                result["message"] = (
                    "The PDF was copied and its ISBN was found. Provide "
                    f"{', '.join(missing_details)} to register it."
                    if self.response_language == "en"
                    else "تم نسخ الملف والعثور على ISBN. أرسل "
                    f"{' و'.join(missing_details)} لتسجيله في الكتالوج."
                )
            else:
                input_labels = {
                    "title": (
                        "Book title: ",
                        "عنوان الكتاب: ",
                    ),
                    "author": (
                        "Book author: ",
                        "مؤلف الكتاب: ",
                    ),
                }
                for field_name in missing_details:
                    english_label, arabic_label = input_labels[field_name]
                    value = input_fn(
                        english_label
                        if self.response_language == "en"
                        else arabic_label
                    ).strip()
                    if not value:
                        result["message"] = (
                            "The PDF was copied but not registered because "
                            f"{field_name} was empty."
                            if self.response_language == "en"
                            else "تم نسخ الملف، لكن لم يُسجّل لأن "
                            f"بيان {field_name} فارغ."
                        )
                        break
                    copy_result[field_name] = value
                    arguments[field_name] = value
                else:
                    try:
                        copy_result["book"] = self.library._register_copied_pdf(
                            copy_result["title"],
                            copy_result["author"],
                            copy_result["isbn"],
                            Path(copy_result["destination"]).name,
                        )
                    except (OSError, ValueError) as error:
                        copy_result["registration_error"] = str(error)
                        result["message"] = (
                            f"The PDF was copied but could not be registered: {error}"
                            if self.response_language == "en"
                            else f"تم نسخ الملف لكن تعذر تسجيله: {error}"
                        )
                    else:
                        copy_result["cataloged"] = True
                        copy_result["needs_book_details"] = []
                        copy_result["message"] = (
                            f"PDF copied and registered with ISBN "
                            f"'{copy_result['isbn']}'."
                        )
                        result["message"] = copy_result["message"]
        if result.get("result") is not None:
            try:
                self.learning_memory.record_command(prompt, tool_name, arguments)
            except (OSError, ValueError) as error:
                result["learning_warning"] = str(error)
        return result

    def _learn_library_books(self):
        try:
            report = self.knowledge_base.learn_library_pdfs()
        except (OSError, ValueError) as error:
            message = (
                f"Could not update the book index: {error}"
                if self.response_language == "en"
                else f"تعذر تحديث فهرس الكتب: {error}"
            )
            print(message)
            return

        if report["indexed"]:
            message = (
                f"Automatically indexed {len(report['indexed'])} new or changed PDFs."
                if self.response_language == "en"
                else f"تمت فهرسة {len(report['indexed'])} ملفات PDF جديدة أو متغيرة تلقائيًا."
            )
            print(message)
        for error in report["errors"]:
            print(f"PDF indexing warning for {error['source']}: {error['error']}")

    def run(self):
        print("MacroSBL local tools agent. Type 'exit' to exit MacroSBL.")
        self._learn_library_books()
        while True:
            prompt = input("You: ").strip()
            if prompt.casefold() == "رجوع":
                return False
            if prompt.casefold() in {"exit", "quit", "خروج"}:
                return True
            try:
                result = self.handle_prompt(prompt, input_fn=input)
            except (OSError, RuntimeError, ValueError) as error:
                if self.response_language == "en":
                    print(f"Unable to complete the request: {error}")
                else:
                    print(f"تعذر تنفيذ الطلب: {error}")
                continue
            output = result.get("output")
            if output:
                print(output)
            if result.get("result") is not None:
                print(json.dumps(result["result"], ensure_ascii=False, indent=2))
            if result.get("message"):
                print(result["message"])
            if result.get("learning_warning"):
                print(f"Learning warning: {result['learning_warning']}")
            if result.get("end_agent"):
                return False
