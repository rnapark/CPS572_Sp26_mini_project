import re

class TuluFilter:
    def __init__(self, max_tokens=300, min_score=2):
        self.max_tokens = max_tokens
        self.min_score = min_score

    def has_code(self, text):
        code_markers = [
            "```python", "```", "import ", "def ",
            "print(", "sympy", "np.", "pd.", "torch.", "```output"
        ]
        return any(marker in text for marker in code_markers)

    def too_long(self, text):
        return len(text.split()) > self.max_tokens

    def has_repetition(self, text):
        lines = text.split("\n")
        return len(lines) != len(set(lines))

    def is_teaching_style(self, text):
        bad_phrases = [
            "let's solve", "step by step", "we will",
            "now we", "let's implement", "we can use python"
        ]
        text = text.lower()
        return any(p in text for p in bad_phrases)

    def is_concise(self, text):
        return len(text.split()) < 150

    def has_clean_final_answer(self, text):
        patterns = [
            r"\\boxed\{.*?\}",
            r"^[-+]?\d+(\.\d+)?$"
        ]
        return any(re.search(p, text.strip()) for p in patterns)

    def has_structure(self, text):
        text = text.strip()
        return (
            text.startswith("{") or
            text.startswith("[") or
            "\n-" in text or
            "\n1." in text
        )

    def has_instruction_constraint(self, prompt):
        keywords = [
            "only", "just", "format", "list",
            "return", "output", "without explanation", "json"
        ]
        prompt = prompt.lower()
        return any(k in prompt for k in keywords)

    def score(self, prompt, response):
        score = 0

        if self.has_code(response): score -= 3
        if self.too_long(response): score -= 2
        if self.is_teaching_style(response): score -= 2
        if self.has_repetition(response): score -= 1

        if self.is_concise(response): score += 2
        if self.has_clean_final_answer(response): score += 2
        if self.has_structure(response): score += 1
        if self.has_instruction_constraint(prompt): score += 2

        return score

    def keep(self, sample):
        prompt = sample["messages"][0]["content"]
        response = sample["messages"][1]["content"]

        s = self.score(prompt, response)
        return s >= self.min_score, s