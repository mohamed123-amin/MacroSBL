import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class LocalTextGenerator:
    def __init__(self, dataset_file=None):
        self.dataset_file = (
            Path(dataset_file)
            if dataset_file
            else Path(__file__).with_name("conversation_dataset.json")
        )
        with self.dataset_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("examples"), list)
            or not data["examples"]
            or any(
                not isinstance(example, dict)
                or not isinstance(example.get("input"), str)
                or not example["input"].strip()
                or not isinstance(example.get("response"), str)
                or not example["response"].strip()
                for example in data["examples"]
            )
        ):
            raise ValueError(
                "The conversation dataset must contain non-empty input/response examples."
            )

        self.examples = data["examples"]
        self.intent_examples = [example["input"] for example in self.examples]
        self._vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=True,
        )
        self._example_matrix = self._vectorizer.fit_transform(self.intent_examples)
        self._generator = None

    def _get_generator(self):
        if self._generator is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as error:
                raise RuntimeError(
                    "Could not import PyTorch/Transformers for text generation: "
                    f"{error}. Install the project requirements in the interpreter "
                    "used to run MacroSBL."
                ) from error
            device = 0 if torch.cuda.is_available() else -1
            self._generator = pipeline(
                "text-generation",
                model=GENERATION_MODEL,
                device=device,
            )
        return self._generator

    def _relevant_examples(self, prompt, limit=2, verified_examples=None):
        examples = self.examples + (verified_examples or [])
        if verified_examples:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                lowercase=True,
            )
            matrix = vectorizer.fit_transform(
                [example["input"] for example in examples]
            )
        else:
            matrix = self._example_matrix
        if verified_examples:
            prompt_vector = vectorizer.transform([prompt])
        elif len("".join(prompt.split())) < 2:
            return []
        else:
            prompt_vector = self._vectorizer.transform([prompt])
        scores = (matrix @ prompt_vector.T).toarray().ravel()
        ranked = sorted(
            (
                (float(score), example)
                for score, example in zip(scores, examples)
                if score > 0
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [example for _, example in ranked[:limit]]

    def matches_dataset_prompt(self, prompt, minimum_score=0.45):
        if len("".join(prompt.split())) < 2:
            return False
        prompt_vector = self._vectorizer.transform([prompt])
        scores = (self._example_matrix @ prompt_vector.T).toarray().ravel()
        return bool(len(scores) and float(scores.max()) >= minimum_score)

    def generate(self, prompt, language="en", verified_examples=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if language not in {"en", "ar"}:
            raise ValueError("Language must be 'en' or 'ar'.")

        language_name = "English" if language == "en" else "Arabic"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are MacroSBL's helpful local assistant. "
                    f"Reply in {language_name}. Answer general questions clearly and "
                    "naturally. Use the examples as style guidance, not as facts for "
                    "unrelated questions. Use the library's PDF tools when a user "
                    "provides a PDF path or asks to learn from a book. Do not claim "
                    "that you read a file unless a PDF tool processed it. If you are "
                    "unsure, say so."
                ),
            }
        ]
        for example in self._relevant_examples(
            prompt,
            verified_examples=verified_examples,
        ):
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["response"]})
        messages.append({"role": "user", "content": prompt.strip()})

        results = self._get_generator()(
            messages,
            max_new_tokens=180,
            do_sample=False,
        )
        if not isinstance(results, list) or not results:
            raise RuntimeError("The text generation model returned no response.")
        generated_text = results[0].get("generated_text")
        if isinstance(generated_text, list):
            answer = next(
                (
                    message.get("content", "").strip()
                    for message in reversed(generated_text)
                    if isinstance(message, dict)
                    and message.get("role") == "assistant"
                    and isinstance(message.get("content"), str)
                    and message["content"].strip()
                ),
                "",
            )
        elif isinstance(generated_text, str):
            answer = generated_text.strip()
        else:
            answer = ""

        if not answer:
            raise RuntimeError("The text generation model returned an empty response.")
        return answer
