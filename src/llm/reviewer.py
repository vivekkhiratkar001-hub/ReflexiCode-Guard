import json
from typing import Any, Dict, List
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.shared.enums import FindingCategory
from src.shared.models import AnalysisFinding, PRContext


class OllamaReviewer:
    """Optional semantic review adapter for a local Ollama generate endpoint."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def review(
        self,
        context: PRContext,
        findings: List[AnalysisFinding],
        repository_rules: str = "",
    ) -> List[AnalysisFinding]:
        prompt = self._build_prompt(context, findings, repository_rules)
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError) as error:
            raise RuntimeError("Could not reach the local Ollama reviewer.") from error
        except json.JSONDecodeError as error:
            raise RuntimeError("Ollama returned an invalid JSON response.") from error

        response_text = body.get("response")
        if not isinstance(response_text, str):
            raise RuntimeError("Ollama response did not contain text.")
        try:
            output = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise RuntimeError("Ollama review output was not valid JSON.") from error
        return findings + self._parse_findings(output)

    @staticmethod
    def _build_prompt(
        context: PRContext,
        findings: List[AnalysisFinding],
        repository_rules: str,
    ) -> str:
        existing = [
            {
                "file_path": item.file_path,
                "line_number": item.line_number,
                "message": item.message,
            }
            for item in findings
        ]
        return (
            "Review this pull request diff for concrete bugs. Do not repeat existing "
            "findings. Return only JSON shaped as "
            '{"findings":[{"file_path":"...","line_number":1,"category":"semantic",'
            '"severity":"low|medium|high|critical","message":"...","evidence":"...",'
            '"suggestion":"..."}]}. Use an empty findings list when there are no '
            "actionable issues.\n"
            f"Title: {context.title}\nDescription: {context.description}\n"
            f"Issue context: {context.issue_context}\n"
            f"Repository rules: {repository_rules}\n"
            f"Existing findings: {json.dumps(existing)}\n"
            f"Diff:\n{context.diff}"
        )

    @staticmethod
    def _parse_findings(output: Any) -> List[AnalysisFinding]:
        if not isinstance(output, dict) or not isinstance(output.get("findings"), list):
            raise RuntimeError("Ollama review output must contain a findings list.")

        parsed: List[AnalysisFinding] = []
        required = (
            "file_path",
            "category",
            "severity",
            "message",
            "evidence",
            "suggestion",
        )
        for item in output["findings"]:
            if not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) for key in required
            ):
                raise RuntimeError("Ollama returned a finding with invalid fields.")
            line_number = item.get("line_number")
            if line_number is not None and (
                not isinstance(line_number, int) or isinstance(line_number, bool)
            ):
                raise RuntimeError("Ollama finding line_number must be an integer or null.")
            parsed.append(
                AnalysisFinding(
                    file_path=item["file_path"],
                    line_number=line_number,
                    category=item["category"] or FindingCategory.SEMANTIC.value,
                    severity=item["severity"],
                    message=item["message"],
                    evidence=item["evidence"],
                    suggestion=item["suggestion"],
                )
            )
        return parsed
