import json
import math
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

if __package__:
    from .sparse_clustering import cluster_sparse_vectors
else:
    from sparse_clustering import cluster_sparse_vectors


class SelfLearningMemory:
    def __init__(self, data_file=None):
        self.data_file = (
            Path(data_file)
            if data_file
            else Path(__file__).with_name("self_learning.json")
        )
        self.data = self._load()
        self._cluster_commands()

    def _load(self):
        if not self.data_file.exists():
            return {
                "commands": [],
                "command_clusters": [],
                "verified_answers": [],
            }

        with self.data_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("The self-learning file must contain an object.")
        for key in ("commands", "verified_answers"):
            if not isinstance(data.get(key, []), list):
                raise ValueError(f"The self-learning '{key}' field must be a list.")

        commands = data.get("commands", [])
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("pattern"), str)
            or not isinstance(item.get("tool"), str)
            for item in commands
        ):
            raise ValueError("Each learned command must have a pattern and tool.")
        verified_answers = data.get("verified_answers", [])
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("question"), str)
            or not isinstance(item.get("answer"), str)
            for item in verified_answers
        ):
            raise ValueError(
                "Each verified answer must have a question and answer."
            )
        return {
            "commands": commands,
            "command_clusters": data.get("command_clusters", []),
            "verified_answers": verified_answers,
        }

    @staticmethod
    def _sanitize_command(prompt, arguments):
        pattern = prompt.casefold().strip()
        values = sorted(
            (
                str(value).casefold()
                for value in arguments.values()
                if value is not None and str(value).strip()
            ),
            key=len,
            reverse=True,
        )
        for value in values:
            pattern = re.sub(re.escape(value), "<value>", pattern)
        pattern = re.sub(r"""['"][^'"]+['"]""", "<value>", pattern)
        pattern = re.sub(r"(?:[a-z]:\\|\\\\)[^\s]+", "<path>", pattern)
        pattern = re.sub(r"\b\d{3,}\b", "<number>", pattern)
        return re.sub(r"\s+", " ", pattern).strip()

    @property
    def commands(self):
        return list(self.data["commands"])

    @property
    def verified_answers(self):
        return list(self.data["verified_answers"])

    @property
    def command_clusters(self):
        return list(self.data["command_clusters"])

    def training_examples(self, limit_per_tool=5):
        counts = Counter()
        examples = []
        for command in reversed(self.data["commands"]):
            tool = command["tool"]
            if counts[tool] >= limit_per_tool:
                continue
            counts[tool] += 1
            examples.append((command["pattern"], tool))
        return examples

    def find_verified_answer(self, question, minimum_score=0.9):
        if len("".join(question.split())) < 4:
            return None

        examples = [
            example
            for example in self.data["verified_answers"]
            if len("".join(example["question"].split())) >= 4
        ]
        if not examples:
            return None

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(
            [example["question"] for example in examples]
        )
        query = vectorizer.transform([question])
        scores = (matrix @ query.T).toarray().ravel()
        best_index = int(scores.argmax())
        if float(scores[best_index]) < minimum_score:
            return None
        return examples[best_index]["answer"]

    def record_command(self, prompt, tool, arguments):
        pattern = self._sanitize_command(prompt, arguments)
        if not pattern:
            return

        existing = next(
            (
                command
                for command in self.data["commands"]
                if command["pattern"] == pattern and command["tool"] == tool
            ),
            None,
        )
        if existing:
            existing["count"] = int(existing.get("count", 1)) + 1
        else:
            self.data["commands"].append(
                {"pattern": pattern, "tool": tool, "count": 1, "cluster": -1}
            )
        self.data["commands"] = self.data["commands"][-500:]
        self._cluster_commands()
        self._save()

    def record_verified_answer(self, question, answer):
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            raise ValueError("A verified answer needs both a question and an answer.")

        normalized_question = question.casefold().rstrip(" .!?؟")
        existing = next(
            (
                item
                for item in self.data["verified_answers"]
                if item["question"].casefold().rstrip(" .!?؟")
                == normalized_question
            ),
            None,
        )
        if existing:
            existing["answer"] = answer
        else:
            self.data["verified_answers"].append(
                {"question": question, "answer": answer}
            )
        self.data["verified_answers"] = self.data["verified_answers"][-200:]
        self._save()

    def _cluster_commands(self):
        commands = self.data["commands"]
        if not commands:
            self.data["command_clusters"] = []
            return

        patterns = [command["pattern"] for command in commands]
        distinct_patterns = len(set(patterns))
        cluster_count = min(
            8,
            distinct_patterns,
            max(1, round(math.sqrt(len(commands)))),
        )
        if cluster_count == 1:
            labels = [0] * len(commands)
        else:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                lowercase=True,
            )
            matrix = vectorizer.fit_transform(patterns)
            labels = cluster_sparse_vectors(matrix, cluster_count)

        grouped = {}
        for command, label in zip(commands, labels):
            cluster = int(label)
            command["cluster"] = cluster
            grouped.setdefault(cluster, []).append(command)

        self.data["command_clusters"] = [
            {
                "id": cluster,
                "size": len(items),
                "tools": dict(Counter(item["tool"] for item in items)),
                "examples": list(dict.fromkeys(item["pattern"] for item in items))[:3],
            }
            for cluster, items in sorted(grouped.items())
        ]

    def _save(self):
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = self.data_file.with_suffix(".tmp")
        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=2)
        temporary_file.replace(self.data_file)
