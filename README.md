# Library data preparation

Install the data-processing dependencies:

```powershell
python -m pip install -r requirements.txt
```

The library application also uses NumPy and pandas to structure, normalize,
and clean the default book catalog created by `create_default_books`. Install
the requirements before running the library or its tests.

When adding a book through `macroSBL.py`, the application trains
`LinearRegression` and `DecisionTreeRegressor` from `isbn_training_data.json`.
Their predicted sequence values are combined, converted to a 13-digit
ISBN-shaped value with a check digit, and checked against the library's existing
numbers. The selected PDF is copied into `library books`, and the ISBN and PDF
filename are saved in `library_data.json`.

The supplied training records contain synthetic ISBN examples. Values generated
by these models are for internal demonstration only; they are not officially
assigned ISBNs and must not be used as publication identifiers.

Prepare clean CSV files from `library_data.json`:

```powershell
python prepare_training_data.py
```

By default, the script creates `training_data` containing:

- `books.csv`: books with a valid title, author, and ISBN, deduplicated by ISBN.
- `members.csv`: members with a valid name and positive integer ID, deduplicated by member ID.
- `loans.csv`: unique loans that reference an exported book and member. A book is kept to at most one active loan.
- `training_data.csv`: one row per book, with availability derived from the cleaned loan records and a `loan_count` feature.

Rows missing required fields are dropped. Optional member activity is set to `True` when missing; unrecognized activity values are dropped. PDF filenames are omitted because they are not useful model features. The source JSON is never overwritten. The script prints counts of dropped, duplicate, and invalid-reference records.

Use custom paths when needed:

```powershell
python prepare_training_data.py --input library_data.json --output-dir training_data
```

## Local natural-language tools agent

Choose **13. Use the local tools agent** from the library menu to ask for book,
member, loan, and PDF operations in Arabic or English, or to chat and ask
general questions. General responses use the local Qwen text-generation model;
PyTorch runs it on CUDA when available and otherwise on CPU. The model is
downloaded on first use and does not call an external API.
`conversation_dataset.json` provides few-shot examples and intent-training
phrases for greetings and common questions; it does not fine-tune model weights.
If the agent cannot extract a required value from a library request, it asks
for that value before calling the tool. Library operations are also exposed as
structured tool schemas through `LocalLibraryAgent.call_tool`.
Say goodbye or type `رجوع` to end the agent session and return to the library
menu; type `exit` to close MacroSBL.

The PDF tools can read selectable text, report PDF metadata and ISBN candidates,
and save extracted text to `pdf_knowledge.json`. Use **learn this PDF** before
asking questions about its contents; searches then retrieve matching text from
that local knowledge file. This is local indexing/retrieval, not fine-tuning a
language model.

The PDF tools accept a path to a PDF or the ISBN of a book already in the
library. Extracted text, metadata, and page text are stored locally in the
project folder. After choosing **learn this PDF**, ask a question about the book in the local
tools agent to get an answer extracted by a multilingual Transformer QA model,
along with its source PDF, page number, and supporting text. Install the
requirements first; the model is downloaded the first time a question is
answered and then reused from the local model cache. This is extractive
question answering, so book answers are grounded in quoted book text rather
than free-form generated explanations. Questions not explicitly about a book
are answered by the general text-generation model. If no relevant book text is
found, the agent reports that instead of inventing a book answer.

### OCR for scanned PDFs

OCR is a manual operation; it is not run during startup or ordinary PDF
indexing. Install its optional Python packages with:

```powershell
python -m pip install -r requirements-ocr.txt
```

These requirements install PyMuPDF for page rendering and EasyOCR for text
recognition. EasyOCR is the only supported OCR engine.

Ask the local tools agent to run OCR on a PDF path or a library-book ISBN.
PyMuPDF renders the pages and EasyOCR recognizes English and Arabic. OCR fills
pages that have no selectable text, then saves the recognized page text to the
local knowledge base so it can be searched and used for book questions. EasyOCR
may download model files on first use.

At agent startup, PDFs directly inside `library books` and `loaned books` are
checked for changes. New or changed files are added to the local knowledge
index, deleted files are removed, and text sections are grouped into topic
clusters using TF-IDF sparse vectors. Completed library-tool requests are
stored as redacted command patterns and grouped by the same lightweight
clustering implementation to help train the local intent classifier. Generated
answers are not learned automatically: say `remember
answer` (or `احفظ الإجابة`) immediately after a response to save it as a
few-shot example for later responses. This local learning does not fine-tune
the language model's weights, and saved data remains in the project folder.
Loaning a book moves its saved PDF from `library books` to `loaned books`;
returning it moves the PDF back. The agent can still locate and index PDFs in
either folder.

The `find_and_copy_pdf` tool searches PDF filenames under the current user's
profile (excluding hidden, system/cache, and linked folders), or accepts an
explicit PDF path inside that profile. It copies a single match into
`library books` without changing the source. It does not scan other Windows
accounts or system-wide folders, overwrite files, or execute operating-system
commands. If several files match, use a more specific filename. When it finds a
valid ISBN in selectable PDF text, it registers the book and links the copied
file to that ISBN. ISBN-10 values are normalized to their equivalent ISBN-13.
If the PDF has an ISBN but lacks title or author metadata, the agent asks for
the missing details before registering it. If no valid ISBN is found, the PDF
is copied but not added to the catalog; use `add_book` to register it manually.
