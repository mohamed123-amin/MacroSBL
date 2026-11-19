import json
from io import StringIO
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from sklearn.feature_extraction.text import TfidfVectorizer

from library_agent import LocalLibraryAgent, PDFKnowledgeBase
from macroSBL import Book, LibraryManagementSystem, Member
from self_learning import SelfLearningMemory
from sparse_clustering import cluster_sparse_vectors
from text_generator import LocalTextGenerator


class LocalLibraryAgentTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        temporary_path = Path(self.temporary_directory.name)
        self.library = LibraryManagementSystem()
        self.library.data_file = temporary_path / "library_data.json"
        self.library.training_file = Path(__file__).with_name("isbn_training_data.json")
        self.library.books_folder = temporary_path / "library books"
        self.library.loaned_books_folder = temporary_path / "loaned books"
        self.library.books = [
            Book("The Atlas of Amber Skies", "Mira Ashford", "9780000000019")
        ]
        self.knowledge = PDFKnowledgeBase(
            self.library,
            data_file=temporary_path / "pdf_knowledge.json",
        )
        self.learning_memory = SelfLearningMemory(
            data_file=temporary_path / "self_learning.json",
        )
        self.agent = LocalLibraryAgent(
            self.library,
            self.knowledge,
            self.learning_memory,
        )

    def test_text_generator_uses_dataset_examples_and_returns_generated_text(self):
        generator = LocalTextGenerator()
        generator._generator = Mock(
            return_value=[
                {
                    "generated_text": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "Hello there!"},
                    ]
                }
            ]
        )

        answer = generator.generate("hello", language="en")

        self.assertEqual(answer, "Hello there!")
        messages = generator._generator.call_args.args[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "hello"})
        self.assertTrue(
            any(message.get("content") == "Hello! How can I help you today?" for message in messages)
        )

    def test_text_generator_includes_explicitly_verified_examples(self):
        generator = LocalTextGenerator()
        generator._generator = Mock(
            return_value=[{"generated_text": "Plants use light."}]
        )

        answer = generator.generate(
            "What is photosynthesis?",
            verified_examples=[
                {
                    "input": "What is photosynthesis?",
                    "response": "Plants use light.",
                }
            ],
        )

        self.assertEqual(answer, "Plants use light.")
        messages = generator._generator.call_args.args[0]
        self.assertIn(
            {"role": "assistant", "content": "Plants use light."},
            messages,
        )

    def test_text_generator_selects_torch_device_for_transformers_pipeline(self):
        for cuda_available, expected_device in ((False, -1), (True, 0)):
            torch_module = ModuleType("torch")
            torch_module.cuda = Mock()
            torch_module.cuda.is_available.return_value = cuda_available
            pipeline = Mock(return_value=Mock())
            transformers_module = ModuleType("transformers")
            transformers_module.pipeline = pipeline
            generator = LocalTextGenerator()

            with patch.dict(
                sys.modules,
                {"torch": torch_module, "transformers": transformers_module},
            ):
                generator._get_generator()

            pipeline.assert_called_once_with(
                "text-generation",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                device=expected_device,
            )

    def test_command_learning_redacts_values_clusters_and_persists_examples(self):
        examples = [
            (
                "search for book Mira Ashford",
                "search_book",
                {"search_term": "Mira Ashford"},
            ),
            (
                "find books by author Lina Corbett",
                "search_book",
                {"search_term": "Lina Corbett"},
            ),
            (
                "loan ISBN 9780000000019 to member 42",
                "loan_book",
                {"isbn": "9780000000019", "member_id": 42},
            ),
        ]
        for prompt, tool, arguments in examples:
            self.learning_memory.record_command(prompt, tool, arguments)

        learned = self.learning_memory.commands
        self.assertTrue(all("Mira Ashford" not in item["pattern"] for item in learned))
        self.assertTrue(all("9780000000019" not in item["pattern"] for item in learned))
        self.assertTrue(self.learning_memory.command_clusters)
        restored = SelfLearningMemory(self.learning_memory.data_file)
        self.assertIn("loan_book", {tool for _, tool in restored.training_examples()})

    def test_short_prompts_do_not_break_text_or_memory_matching(self):
        self.assertFalse(self.agent.text_generator.matches_dataset_prompt("a"))
        self.assertIsNone(self.learning_memory.find_verified_answer("hi"))

    def test_sparse_clustering_groups_similar_texts_without_sklearn_cluster(self):
        matrix = TfidfVectorizer().fit_transform(
            [
                "library checkout borrowing",
                "library lending return",
                "orbital planet astronomy",
                "planet orbit astronomy",
            ]
        )

        labels = cluster_sparse_vectors(matrix, 2)

        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_only_explicitly_remembered_generated_answers_are_reused(self):
        self.agent.text_generator.generate = Mock(return_value="Plants use light.")
        prompt = "What is photosynthesis?"

        generated = self.agent.handle_prompt(prompt)
        self.assertEqual(generated["message"], "Plants use light.")
        self.assertEqual(self.learning_memory.verified_answers, [])

        remembered = self.agent.handle_prompt("remember answer")
        repeated = self.agent.handle_prompt(prompt)

        self.assertIn("Saved", remembered["message"])
        self.assertEqual(repeated["tool"], "verified_memory")
        self.assertEqual(repeated["message"], "Plants use light.")
        self.agent.text_generator.generate.assert_called_once()

    def test_library_pdf_scan_indexes_new_and_changed_files_only(self):
        books_folder = self.library.books_folder
        books_folder.mkdir()
        pdf_path = books_folder / "Sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")
        with (
            patch.object(self.knowledge, "learn_pdf", return_value={}) as learn_pdf,
            patch.object(
                self.knowledge,
                "_discover_library_topics",
                return_value=[{"cluster": 0, "keywords": ["sample"]}],
            ),
        ):
            first = self.knowledge.learn_library_pdfs()
            second = self.knowledge.learn_library_pdfs()
            pdf_path.write_bytes(b"%PDF-1.5 changed")
            third = self.knowledge.learn_library_pdfs()

        self.assertEqual(len(first["indexed"]), 1)
        self.assertEqual(len(second["unchanged"]), 1)
        self.assertEqual(len(third["indexed"]), 1)
        self.assertEqual(learn_pdf.call_count, 2)

    def test_library_pdf_scan_tracks_files_moved_to_loaned_folder(self):
        self.library.books_folder.mkdir()
        self.library.loaned_books_folder.mkdir()
        source = self.library.books_folder / "Sample.pdf"
        loaned = self.library.loaned_books_folder / source.name
        source.write_bytes(b"%PDF-1.4")
        with (
            patch.object(self.knowledge, "learn_pdf", return_value={}) as learn_pdf,
            patch.object(
                self.knowledge,
                "_discover_library_topics",
                return_value=[],
            ),
        ):
            first = self.knowledge.learn_library_pdfs()
            source.replace(loaned)
            second = self.knowledge.learn_library_pdfs()

        self.assertEqual(first["indexed"], [str(source.resolve())])
        self.assertEqual(second["indexed"], [str(loaned.resolve())])
        self.assertEqual(second["removed"], [str(source.resolve())])
        self.assertEqual(learn_pdf.call_count, 2)

    def test_infers_arabic_search_tool_and_extracts_search_term(self):
        result = self.agent.infer_tool("ابحث عن كتاب Mira Ashford")

        self.assertEqual(result["tool"], "search_book")
        self.assertEqual(result["arguments"]["search_term"], "Mira Ashford")

    def test_tool_call_returns_structured_book_records(self):
        result = self.agent.call_tool("see_all_books", {})

        self.assertEqual(result["tool"], "see_all_books")
        self.assertEqual(result["result"][0]["title"], "The Atlas of Amber Skies")
        self.assertIn("All books", result["output"])

    def test_missing_tool_arguments_are_reported_without_execution(self):
        result = self.agent.handle_prompt("add a book")

        self.assertEqual(result["tool"], "add_book")
        self.assertCountEqual(result["missing_arguments"], ["title", "author", "pdf_path"])

    def test_tool_call_rejects_unknown_or_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "Unexpected arguments"):
            self.agent.call_tool("search_book", {"search_term": "Atlas", "other": 1})
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.agent.call_tool("remove_member", {"member_id": True})

    def test_add_member_tool_saves_member_and_returns_result(self):
        result = self.agent.call_tool("add_member", {"name": "Reader One"})

        self.assertEqual(result["result"]["member_id"], 1)
        self.assertTrue(self.library.data_file.is_file())
        saved = json.loads(self.library.data_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["members"][0]["name"], "Reader One")

    def test_find_and_copy_pdf_searches_user_folders_and_preserves_original(self):
        search_root = Path(self.temporary_directory.name) / "user profile"
        source = search_root / "Downloads" / "Stewart Main Book.pdf"
        excluded_match = search_root / "AppData" / "Stewart Hidden Copy.pdf"
        source.parent.mkdir(parents=True)
        excluded_match.parent.mkdir(parents=True)
        source.write_bytes(b"%PDF-1.4 book")
        excluded_match.write_bytes(b"%PDF-1.4 private")

        result = self.library.find_and_copy_pdf("Stewart", search_root=search_root)

        destination = self.library.books_folder / source.name
        self.assertTrue(result["copied"])
        self.assertEqual(result["source"], str(source.resolve()))
        self.assertEqual(result["destination"], str(destination))
        self.assertEqual(source.read_bytes(), b"%PDF-1.4 book")
        self.assertEqual(destination.read_bytes(), b"%PDF-1.4 book")
        self.assertIsNone(result["isbn"])
        self.assertFalse(result["cataloged"])

    def test_find_and_copy_pdf_registers_valid_extracted_isbn(self):
        search_root = Path(self.temporary_directory.name) / "user profile"
        source = search_root / "Downloads" / "Atlas.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"%PDF-1.4 sample")
        with patch.object(
            self.library,
            "_extract_pdf_catalog_details",
            return_value={
                "title": "The Atlas",
                "author": "Mira Ashford",
                "isbn": "9780306406157",
            },
        ):
            result = self.library.find_and_copy_pdf(
                "Atlas",
                search_root=search_root,
            )

        self.assertTrue(result["copied"])
        self.assertTrue(result["cataloged"])
        self.assertEqual(result["isbn"], "9780306406157")
        self.assertEqual(result["book"]["title"], "The Atlas")
        self.assertEqual(
            result["book"]["pdf_filename"],
            "Atlas.pdf",
        )
        saved = json.loads(self.library.data_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["books"][-1]["isbn"], "9780306406157")

    def test_find_and_copy_pdf_converts_valid_isbn10_to_isbn13(self):
        self.assertEqual(
            self.library._normalize_isbn_candidate("0-306-40615-2"),
            "9780306406157",
        )
        self.assertIsNone(
            self.library._normalize_isbn_candidate("9780306406158")
        )

    def test_pdf_catalog_details_extract_metadata_and_valid_isbn(self):
        page = Mock()
        page.extract_text.return_value = "ISBN 0-306-40615-2"
        reader = Mock(
            metadata={"/Title": "The Atlas", "/Author": "Mira Ashford"},
            pages=[page],
        )

        with patch("pypdf.PdfReader", return_value=reader) as pdf_reader:
            details = self.library._extract_pdf_catalog_details("book.pdf")

        pdf_reader.assert_called_once_with("book.pdf", strict=False)
        self.assertEqual(
            details,
            {
                "title": "The Atlas",
                "author": "Mira Ashford",
                "isbn": "9780306406157",
            },
        )

    def test_find_and_copy_pdf_requires_a_unique_match(self):
        search_root = Path(self.temporary_directory.name) / "user profile"
        first = search_root / "Downloads" / "Stewart First.pdf"
        second = search_root / "Documents" / "Stewart Second.pdf"
        first.parent.mkdir(parents=True)
        second.parent.mkdir(parents=True)
        first.write_bytes(b"%PDF-1.4 first")
        second.write_bytes(b"%PDF-1.4 second")

        result = self.library.find_and_copy_pdf("Stewart", search_root=search_root)

        self.assertTrue(result["found"])
        self.assertFalse(result["copied"])
        self.assertEqual(len(result["matches"]), 2)
        self.assertFalse(self.library.books_folder.exists())

    def test_find_and_copy_pdf_accepts_an_explicit_path(self):
        profile_root = Path(self.temporary_directory.name) / "user profile"
        source = profile_root / "L07_assignment.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"%PDF-1.4 assignment")

        with patch("macroSBL.Path.home", return_value=profile_root):
            result = self.library.find_and_copy_pdf(pdf_path=str(source))

        self.assertTrue(result["copied"])
        self.assertEqual(
            (self.library.books_folder / source.name).read_bytes(),
            b"%PDF-1.4 assignment",
        )

    def test_agent_prompts_for_missing_book_details_before_registration(self):
        profile_root = Path(self.temporary_directory.name) / "user profile"
        source = profile_root / "Downloads" / "Book without metadata.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"%PDF-1.4 sample")
        with (
            patch("macroSBL.Path.home", return_value=profile_root),
            patch.object(
                self.library,
                "_extract_pdf_catalog_details",
                return_value={
                    "title": None,
                    "author": "Mira Ashford",
                    "isbn": "9780306406157",
                },
            ),
        ):
            input_fn = Mock(return_value="The Atlas")
            result = self.agent.handle_prompt(
                f"please add {source.name} with path {source}",
                input_fn=input_fn,
            )

        input_fn.assert_called_once_with("Book title: ")
        self.assertTrue(result["result"]["cataloged"])
        self.assertEqual(result["result"]["book"]["title"], "The Atlas")
        self.assertEqual(result["result"]["book"]["author"], "Mira Ashford")

    def test_agent_extracts_search_term_for_pdf_copy_request(self):
        for prompt in (
            "ابحث عن ملف PDF باسم Stewart وانسخه للمكتبة",
            "search my folders for the PDF named Stewart and copy it to the library",
            "find L07_assignment.pdf",
            "ابحث عن L07_assignment.pdf",
        ):
            with self.subTest(prompt=prompt):
                result = self.agent.infer_tool(prompt)

                self.assertEqual(result["tool"], "find_and_copy_pdf")
                expected_term = (
                    "Stewart" if "Stewart" in prompt else "L07_assignment.pdf"
                )
                self.assertEqual(result["arguments"]["search_term"], expected_term)

        pdf_path = str(Path.home() / "project3" / "L07_assignment.pdf")
        result = self.agent.infer_tool(
            f"please add L07_assignment.pdf with path {pdf_path}"
        )
        self.assertEqual(result["tool"], "find_and_copy_pdf")
        self.assertEqual(result["arguments"]["pdf_path"], pdf_path)

        for prompt in (
            "please add L07_assignment.pdf",
            "please add L07_assignment.pdf to the library",
        ):
            result = self.agent.infer_tool(prompt)
            self.assertEqual(result["tool"], "find_and_copy_pdf")
            self.assertEqual(result["arguments"]["search_term"], "L07_assignment.pdf")

        catalog_prompt = (
            f"add a book title: Calculus author: James Stewart pdf_path: {pdf_path}"
        )
        result = self.agent.infer_tool(catalog_prompt)
        self.assertEqual(result["tool"], "add_book")

    def test_agent_startup_does_not_print_discovered_topics(self):
        self.knowledge.learn_library_pdfs = Mock(
            return_value={
                "indexed": [],
                "errors": [],
                "topics": [{"keywords": ["sample"], "chunk_count": 1}],
            }
        )
        with patch("sys.stdout", new_callable=StringIO) as output:
            self.agent._learn_library_books()

        self.assertEqual(output.getvalue(), "")

    def test_book_member_and_loan_tools_run_without_interactive_input(self):
        member = self.agent.call_tool("add_member", {"name": "Reader One"})["result"]
        isbn = self.library.books[0].isbn

        loan = self.agent.call_tool(
            "loan_book",
            {"member_id": member["member_id"], "isbn": isbn},
        )
        self.assertTrue(loan["result"]["loaned"])
        self.assertFalse(self.library.books[0].available)

        returned = self.agent.call_tool(
            "return_book",
            {"member_id": member["member_id"], "isbn": isbn},
        )
        self.assertTrue(returned["result"]["returned"])
        self.assertTrue(self.library.books[0].available)

        modified = self.agent.call_tool(
            "modify_book",
            {
                "isbn": isbn,
                "new_title": "A New Atlas",
                "new_author": "Mira Ashford",
            },
        )
        self.assertEqual(modified["result"]["title"], "A New Atlas")

        removed = self.agent.call_tool("remove_book", {"isbn": isbn})
        self.assertTrue(removed["result"]["removed"])
        self.assertEqual(self.agent.call_tool("see_all_books", {})["result"], [])

    def test_loan_moves_pdf_and_return_restores_it_for_agent_tool_calls(self):
        self.library.books_folder.mkdir()
        filename = self.library.make_pdf_filename(
            self.library.books[0].title,
            self.library.books[0].isbn,
        )
        self.library.books[0].pdf_filename = filename
        source = self.library.books_folder / filename
        loaned_pdf = self.library.loaned_books_folder / filename
        source.write_bytes(b"%PDF-1.4 sample")
        member = self.agent.call_tool(
            "add_member",
            {"name": "Reader One"},
        )["result"]

        loan = self.agent.call_tool(
            "loan_book",
            {"member_id": member["member_id"], "isbn": self.library.books[0].isbn},
        )

        self.assertTrue(loan["result"]["loaned"])
        self.assertFalse(source.exists())
        self.assertTrue(loaned_pdf.is_file())
        self.assertEqual(self.library.get_pdf_path(filename), loaned_pdf)
        self.assertEqual(
            self.knowledge._resolve_pdf(isbn=self.library.books[0].isbn),
            loaned_pdf.resolve(),
        )

        reloaded_library = LibraryManagementSystem()
        reloaded_library.data_file = self.library.data_file
        reloaded_library.training_file = self.library.training_file
        reloaded_library.books_folder = self.library.books_folder
        reloaded_library.loaned_books_folder = self.library.loaned_books_folder
        self.assertTrue(reloaded_library.load_data())
        self.assertFalse(reloaded_library.books[0].available)
        self.assertEqual(reloaded_library.get_pdf_path(filename), loaned_pdf)

        returned = self.agent.call_tool(
            "return_book",
            {"member_id": member["member_id"], "isbn": self.library.books[0].isbn},
        )

        self.assertTrue(returned["result"]["returned"])
        self.assertTrue(source.is_file())
        self.assertFalse(loaned_pdf.exists())

    def test_loan_does_not_overwrite_existing_loaned_pdf(self):
        self.library.books_folder.mkdir()
        self.library.loaned_books_folder.mkdir()
        filename = self.library.make_pdf_filename(
            self.library.books[0].title,
            self.library.books[0].isbn,
        )
        self.library.books[0].pdf_filename = filename
        source = self.library.books_folder / filename
        destination = self.library.loaned_books_folder / filename
        source.write_bytes(b"library copy")
        destination.write_bytes(b"existing loaned copy")
        self.library.members.append(Member("Reader One", 1))

        result = self.library.loan_book(1, self.library.books[0].isbn)

        self.assertIsNone(result)
        self.assertEqual(source.read_bytes(), b"library copy")
        self.assertEqual(destination.read_bytes(), b"existing loaned copy")
        self.assertEqual(self.library.loans, [])
        self.assertTrue(self.library.books[0].available)

    def test_failed_loan_save_restores_pdf_to_library_folder(self):
        self.library.books_folder.mkdir()
        filename = self.library.make_pdf_filename(
            self.library.books[0].title,
            self.library.books[0].isbn,
        )
        self.library.books[0].pdf_filename = filename
        source = self.library.books_folder / filename
        source.write_bytes(b"%PDF-1.4 sample")
        self.library.members.append(Member("Reader One", 1))

        with patch.object(self.library, "save_data", side_effect=OSError("disk full")):
            result = self.library.loan_book(1, self.library.books[0].isbn)

        self.assertIsNone(result)
        self.assertTrue(source.is_file())
        self.assertFalse((self.library.loaned_books_folder / filename).exists())
        self.assertEqual(self.library.loans, [])
        self.assertTrue(self.library.books[0].available)

    def test_pdf_tool_reads_existing_pdf(self):
        pdf_path = Path(__file__).with_name("Problem_Set1.pdf")
        result = self.agent.call_tool("analyze_pdf", {"pdf_path": str(pdf_path)})

        self.assertGreater(result["result"]["page_count"], 0)
        self.assertEqual(result["tool"], "analyze_pdf")

    def test_learn_pdf_persists_text_for_local_search(self):
        extracted = {
            "path": str(Path(self.temporary_directory.name) / "source.pdf"),
            "page_count": 1,
            "metadata": {"title": "Sample", "author": None, "subject": None},
            "pages": [
                {
                    "page": 1,
                    "text": "The greenhouse effect traps heat in the atmosphere.",
                }
            ],
            "text": "The greenhouse effect traps heat in the atmosphere.",
        }
        with patch.object(self.knowledge, "extract", return_value=extracted):
            learned = self.knowledge.learn_pdf(pdf_path="source.pdf")

        self.assertEqual(learned["characters_indexed"], len(extracted["text"]))
        saved = json.loads(self.knowledge.data_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["documents"][0]["text"], extracted["text"])
        self.assertEqual(saved["documents"][0]["pages"], extracted["pages"])
        results = self.knowledge.search("greenhouse effect")
        self.assertEqual(len(results), 1)
        self.assertIn("greenhouse effect", results[0]["text"])
        self.assertEqual(results[0]["page"], 1)

    def test_ocr_pdf_recognizes_scanned_pages_and_indexes_the_text(self):
        image = Mock()
        extracted = {
            "path": str(Path(self.temporary_directory.name) / "scanned.pdf"),
            "page_count": 2,
            "metadata": {"title": "Sample", "author": None, "subject": None},
            "pages": [
                {"page": 1, "text": "Selectable text."},
                {"page": 2, "text": ""},
            ],
            "text": "Selectable text.",
        }
        with (
            patch.object(self.knowledge, "extract", return_value=extracted),
            patch.object(
                self.knowledge,
                "_render_pdf_pages",
                return_value=((2, image) for _ in [None]),
            ) as render_pages,
            patch.object(
                self.knowledge,
                "_recognize_image",
                return_value="Text recognized by OCR.",
            ) as recognize,
        ):
            result = self.knowledge.ocr_pdf(pdf_path="scanned.pdf")

        render_pages.assert_called_once_with(extracted["path"], [2])
        recognize.assert_called_once_with(image)
        image.close.assert_called_once()
        self.assertEqual(result["engine"], "easyocr")
        self.assertEqual(result["ocr_pages"], [2])
        saved = json.loads(self.knowledge.data_file.read_text(encoding="utf-8"))
        self.assertEqual(
            saved["documents"][0]["pages"][1]["text"],
            "Text recognized by OCR.",
        )
        self.assertEqual(
            saved["documents"][0]["text"],
            "Selectable text.\nText recognized by OCR.",
        )

    def test_pdf_page_renderer_uses_pymupdf_and_closes_the_document(self):
        document = Mock()
        page = document.load_page.return_value
        page.get_pixmap.return_value = Mock(
            width=1,
            height=1,
            samples=bytes((255, 0, 0)),
        )
        pymupdf = ModuleType("pymupdf")
        pymupdf.open = Mock(return_value=document)
        pymupdf.Matrix = Mock(return_value="scale")
        pymupdf.csRGB = "rgb"

        with patch.dict(sys.modules, {"pymupdf": pymupdf}):
            rendered_pages = self.knowledge._render_pdf_pages("source.pdf", [1])
            page_number, image = next(rendered_pages)
            self.assertEqual(page_number, 1)
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            image.close()
            with self.assertRaises(StopIteration):
                next(rendered_pages)

        pymupdf.open.assert_called_once_with("source.pdf")
        pymupdf.Matrix.assert_called_once_with(2, 2)
        page.get_pixmap.assert_called_once_with(
            matrix="scale",
            colorspace="rgb",
            alpha=False,
        )
        document.close.assert_called_once()

    def test_ocr_pdf_rejects_empty_results_without_saving(self):
        image = Mock()
        extracted = {
            "path": str(Path(self.temporary_directory.name) / "scanned.pdf"),
            "page_count": 1,
            "metadata": {"title": None, "author": None, "subject": None},
            "pages": [{"page": 1, "text": ""}],
            "text": "",
        }
        with (
            patch.object(self.knowledge, "extract", return_value=extracted),
            patch.object(
                self.knowledge,
                "_render_pdf_pages",
                return_value=((1, image) for _ in [None]),
            ),
            patch.object(self.knowledge, "_recognize_image", return_value="  "),
            self.assertRaisesRegex(ValueError, "did not recognize any text"),
        ):
            self.knowledge.ocr_pdf(pdf_path="scanned.pdf")

        self.assertFalse(self.knowledge.data_file.exists())
        image.close.assert_called_once()

    def test_ocr_request_routes_to_easyocr_and_pdf_path(self):
        pdf_path = r"C:\Users\Ahmed.Amin\Downloads\scanned.pdf"
        inferred = self.agent.infer_tool(
            f'Run OCR on this scanned PDF using EasyOCR at "{pdf_path}"'
        )

        self.assertEqual(inferred["tool"], "ocr_pdf")
        self.assertEqual(inferred["arguments"], {"pdf_path": pdf_path})

    def test_pdf_question_answer_includes_page_and_supporting_text(self):
        extracted = {
            "path": str(Path(self.temporary_directory.name) / "source.pdf"),
            "page_count": 1,
            "metadata": {"title": "Sample", "author": None, "subject": None},
            "pages": [
                {
                    "page": 4,
                    "text": "The greenhouse effect traps heat in the atmosphere.",
                }
            ],
            "text": "The greenhouse effect traps heat in the atmosphere.",
        }
        with patch.object(self.knowledge, "extract", return_value=extracted):
            self.knowledge.learn_pdf(pdf_path="source.pdf")

        with patch.object(
            self.knowledge,
            "_get_qa_pipeline",
            return_value=Mock(
                return_value={"answer": "traps heat", "score": 0.92}
            ),
        ):
            result = self.knowledge.answer_pdf_question(
                "What does the greenhouse effect do?"
            )

        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "traps heat")
        self.assertEqual(result["sources"][0]["page"], 4)
        self.assertIn("traps heat", result["sources"][0]["evidence"])

    def test_pdf_question_reports_when_no_pdf_has_been_learned(self):
        result = self.knowledge.answer_pdf_question("What is in this book?")

        self.assertFalse(result["found"])
        self.assertEqual(result["sources"], [])

    def test_question_prompt_is_extracted_for_pdf_answering_tool(self):
        inferred = self.agent.infer_tool(
            "جاوب على سؤالي من الكتاب: ما سبب الاحتباس الحراري؟"
        )

        self.assertEqual(inferred["tool"], "answer_pdf_question")
        self.assertIn("الاحتباس الحراري", inferred["arguments"]["question"])

    def test_agent_reports_runtime_errors_and_continues_running(self):
        with (
            patch("builtins.input", side_effect=["ask a question", "exit"]),
            patch.object(
                self.agent,
                "handle_prompt",
                side_effect=RuntimeError("Question-answering model is unavailable."),
            ),
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            should_exit = self.agent.run()

        self.assertIn(
            "تعذر تنفيذ الطلب: Question-answering model is unavailable.",
            output.getvalue(),
        )
        self.assertTrue(should_exit)

    def test_agent_exit_signal_stops_the_library_main_loop(self):
        self.library.load_data = Mock(return_value=True)
        agent = Mock()
        agent.run.return_value = True
        with (
            patch("builtins.input", side_effect=["13"]),
            patch("library_agent.LocalLibraryAgent", return_value=agent),
            patch("sys.stdout", new_callable=StringIO),
        ):
            self.library.run()

        agent.run.assert_called_once()

    def test_return_command_goes_back_to_library_menu(self):
        with patch("builtins.input", return_value="رجوع"):
            should_exit = self.agent.run()

        self.assertFalse(should_exit)

    def test_farewell_replies_and_ends_the_agent_session(self):
        with (
            patch("builtins.input", side_effect=["no thank farewell", "no"]) as user_input,
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            should_exit = self.agent.run()

        self.assertFalse(should_exit)
        self.assertEqual(user_input.call_count, 1)
        self.assertIn("Goodbye!", output.getvalue())

    def test_agent_handles_greetings_and_remembers_english_preference(self):
        self.agent.text_generator.generate = Mock(return_value="Generated response.")
        language_change = self.agent.handle_prompt("speak to me in english please")
        greeting = self.agent.handle_prompt("hello")
        answer = self.agent.handle_prompt("what is your name?")

        self.assertEqual(greeting["message"], "Generated response.")
        self.assertIn("I'll speak English", language_change["message"])
        self.assertEqual(answer["tool"], "general_chat")
        self.assertEqual(answer["message"], "Generated response.")
        self.assertEqual(
            self.agent.text_generator.generate.call_args.kwargs["language"],
            "en",
        )

    def test_general_question_uses_input_language_without_preference(self):
        self.agent.text_generator.generate = Mock(return_value="إجابة")

        self.agent.handle_prompt("ما اسمك؟")

        self.assertEqual(
            self.agent.text_generator.generate.call_args.kwargs["language"],
            "ar",
        )

    def test_exact_general_question_uses_text_generation(self):
        self.agent.text_generator.generate = Mock(return_value="I am the assistant.")

        result = self.agent.handle_prompt("what is your name?")

        self.assertEqual(result["tool"], "general_chat")
        self.assertEqual(result["message"], "I am the assistant.")

    def test_read_request_with_pdf_path_routes_to_learning_tool(self):
        pdf_path = r"C:\Users\Ahmed.Amin\Downloads\Stewart Main Book.pdf"
        prompt = (
            "hello macrosbl read the book of stewart with the path "
            f'"{pdf_path}"'
        )
        with patch.object(
            self.agent,
            "call_tool",
            return_value={"tool": "learn_pdf", "result": {}, "output": ""},
        ) as call_tool:
            result = self.agent.handle_prompt(prompt)

        self.assertEqual(result["tool"], "learn_pdf")
        call_tool.assert_called_once_with("learn_pdf", {"pdf_path": pdf_path})


if __name__ == "__main__":
    unittest.main()
