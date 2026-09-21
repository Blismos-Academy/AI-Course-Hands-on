import base64
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Tuple


logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    allowed: bool
    question: str
    reason: str = ""
    pii_detected: bool = False
    injection_detected: bool = False


class AgentSafety:
    """
    Input and output safety layer for the agentic system.

    Protection layers:
    1. Input validation / size limits
    2. Unicode normalization
    3. Dangerous-request detection
    4. PII / secret detection and redaction
    5. Prompt-injection / jailbreak detection
    6. Encoded prompt-injection detection
    7. Dangerous-command detection
    8. Output PII redaction

    Important:
    Blocked requests must never reach:
        Safety -> Memory -> Planner -> Tools/RAG/MCP -> LLM
    """

    MAX_INPUT_LENGTH = 4000

    # ---------------------------------------------------------
    # PII / SECRET PATTERNS
    # ---------------------------------------------------------

    PII_PATTERNS = {
        "email": re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),

        # Indian mobile number
        "phone": re.compile(
            r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
        ),

        # Aadhaar
        "aadhaar": re.compile(
            r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"
        ),

        # PAN
        "pan": re.compile(
            r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])",
            re.IGNORECASE,
        ),

        # Credit/debit card-like numbers
        "credit_card": re.compile(
            r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"
        ),

        # Common API keys
        "api_key": re.compile(
            r"\b(?:"
            r"sk-[A-Za-z0-9_-]{16,}"
            r"|gsk_[A-Za-z0-9_-]{16,}"
            r"|AIza[A-Za-z0-9_-]{20,}"
            r")\b"
        ),

        # password=xxx / api_key=xxx / secret=xxx
        "password": re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_ -]?key)"
            r"\s*[:=]\s*\S+"
        ),
    }

    # ---------------------------------------------------------
    # PROMPT INJECTION / JAILBREAK
    # ---------------------------------------------------------

    INJECTION_PATTERNS = [
        r"\bignore\s+(all|any|the|previous|prior)\s+(instructions?|rules?)\b",
        r"\bdisregard\s+(all|any|the|previous|prior)\s+(instructions?|rules?)\b",
        r"\bforget\s+(all|your|the)\s+(instructions?|rules?|system\s+prompt)\b",
        r"\b(system|developer)\s+prompt\b",
        r"\breveal\s+(your|the)\s+(system|developer)\s+prompt\b",
        r"\bshow\s+(me\s+)?(your|the)\s+(system|developer)\s+prompt\b",
        r"\bprint\s+(your|the)\s+(system|developer)\s+message\b",
        r"\bjailbreak\b",
        r"\bdo\s+anything\s+now\b",
        r"\bDAN\b",
        r"\bbypass\s+(safety|security|guardrails?|restrictions?)\b",
        r"\bdisable\s+(safety|security|guardrails?|filters?)\b",
        r"\bdeveloper\s+mode\b",
        r"\bact\s+as\s+(an?\s+)?unrestricted\b",
        r"\bignore\s+tool\s+restrictions\b",
        r"\breveal\s+(secrets?|keys?|credentials?|tokens?)\b",
        r"\bexecute\s+(arbitrary|shell|system)\s+(command|code)\b",
    ]

    # ---------------------------------------------------------
    # DANGEROUS COMMANDS
    # ---------------------------------------------------------

    DANGEROUS_COMMAND_PATTERNS = [
        r"\b(?:rm|del|erase|format)\s+[-/\\]",
        r"\b(?:powershell|cmd|bash|sh)\b.*\b(?:-enc|-command|exec|invoke)\b",
        r"\bdelete\s+(all|every|everything)\s+(file|data|record)s?\b",
        r"\bexfiltrat\w*\b",
        r"\bsteal\s+(?:passwords?|tokens?|credentials?|keys?)\b",
    ]

    # ---------------------------------------------------------
    # DANGEROUS PHYSICAL-HARM REQUESTS
    # ---------------------------------------------------------

    DANGEROUS_REQUEST_PATTERNS = [
        # Bomb / explosive creation
        r"\bhow\s+to\s+(?:make|build|create)\s+(?:a\s+|an\s+)?bomb\b",
        r"\bhow\s+to\s+(?:make|build|create)\s+(?:a\s+|an\s+)?explosive\w*\b",
        r"\b(?:make|build|create)\s+(?:a\s+|an\s+)?explosive\w*\b",

        # Killing / murder
        r"\bhow\s+to\s+(?:kill|murder|assassinate)\b",
        r"\b(?:kill|murder|assassinate)\s+(?:a\s+|an\s+)?"
        r"(?:person|people|someone|somebody)\b",

        # Physical attacks
        r"\bhow\s+to\s+(?:hurt|attack|injure)\s+"
        r"(?:a\s+|an\s+)?(?:person|people|someone|somebody)\b",
    ]

    # ---------------------------------------------------------
    # MAIN SAFETY CHECK
    # ---------------------------------------------------------

    def check(self, question: str) -> SafetyResult:

        # 1. Validate input type
        if not isinstance(question, str):
            return SafetyResult(
                False,
                "",
                "Invalid input type.",
            )

        # 2. Normalize Unicode
        question = unicodedata.normalize(
            "NFKC",
            question,
        ).strip()

        # 3. Empty input
        if not question:
            return SafetyResult(
                False,
                "",
                "Question cannot be empty.",
            )

        # 4. Input size limit
        if len(question) > self.MAX_INPUT_LENGTH:
            return SafetyResult(
                False,
                "",
                f"Input exceeds the {self.MAX_INPUT_LENGTH}-character limit.",
            )

        # -----------------------------------------------------
        # IMPORTANT:
        # Dangerous requests are checked BEFORE planner/LLM.
        # -----------------------------------------------------

        if self._has_dangerous_request(question):
            return SafetyResult(
                False,
                "",
                "Dangerous request involving serious physical harm detected.",
            )

        # 5. Prompt injection / jailbreak
        if self._has_injection(question):
            return SafetyResult(
                False,
                "",
                "Prompt injection or jailbreak attempt detected.",
                injection_detected=True,
            )

        # 6. Dangerous system commands
        if self._has_dangerous_command(question):
            return SafetyResult(
                False,
                "",
                "Unsupported or dangerous command detected.",
            )

        # 7. PII / secret redaction
        safe_question, pii_detected = self._redact_pii(question)

        # 8. Allow request
        return SafetyResult(
            True,
            safe_question,
            reason="Input passed safety checks.",
            pii_detected=pii_detected,
        )

    # ---------------------------------------------------------
    # DANGEROUS REQUEST DETECTION
    # ---------------------------------------------------------

    def _has_dangerous_request(self, text: str) -> bool:

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in self.DANGEROUS_REQUEST_PATTERNS
        )

    # ---------------------------------------------------------
    # PROMPT INJECTION DETECTION
    # ---------------------------------------------------------

    def _has_injection(self, text: str) -> bool:

        lowered = text.lower()

        # Direct injection patterns
        for pattern in self.INJECTION_PATTERNS:

            if re.search(
                pattern,
                lowered,
                re.IGNORECASE,
            ):
                return True

        # Base64 encoded injection detection
        tokens = re.findall(
            r"(?<![A-Za-z0-9+/])"
            r"[A-Za-z0-9+/]{24,}={0,2}"
            r"(?![A-Za-z0-9+/])",
            text,
        )

        for token in tokens:

            try:

                decoded = base64.b64decode(
                    token,
                    validate=True,
                ).decode(
                    "utf-8",
                    errors="ignore",
                )

            except Exception:
                continue

            if decoded and any(
                re.search(
                    pattern,
                    decoded,
                    re.IGNORECASE,
                )
                for pattern in self.INJECTION_PATTERNS
            ):
                return True

        return False

    # ---------------------------------------------------------
    # DANGEROUS COMMAND DETECTION
    # ---------------------------------------------------------

    def _has_dangerous_command(self, text: str) -> bool:

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in self.DANGEROUS_COMMAND_PATTERNS
        )

    # ---------------------------------------------------------
    # PII REDACTION
    # ---------------------------------------------------------

    def _redact_pii(self, text: str) -> Tuple[str, bool]:

        pii_detected = False
        safe_text = text

        for pii_type, pattern in self.PII_PATTERNS.items():

            safe_text, count = pattern.subn(
                f"[REDACTED_{pii_type.upper()}]",
                safe_text,
            )

            if count:
                pii_detected = True

        return safe_text, pii_detected

    # ---------------------------------------------------------
    # OUTPUT SAFETY
    # ---------------------------------------------------------

    def sanitize_output(self, text: str) -> Tuple[str, bool]:
        """
        Final output guard.

        Removes PII/secrets accidentally generated by:
        - LLM
        - tools
        - MCP
        - memory
        - RAG
        """

        if not isinstance(text, str):
            text = str(text)

        safe_text, pii_detected = self._redact_pii(text)

        return safe_text, pii_detected

    def output_safe_message(self, text: str) -> str:
        """
        Return a safe user-facing response.
        """

        safe_text, _ = self.sanitize_output(text)

        return safe_text

    # ---------------------------------------------------------
    # BLOCKED REQUEST MESSAGE
    # ---------------------------------------------------------

    def safe_message(self, result: SafetyResult) -> str:

        if "dangerous request" in result.reason.lower():

            return (
                "I can't help with instructions for making explosives "
                "or weapons or for seriously harming someone."
            )

        if result.injection_detected:

            return (
                "I can't process that request because it contains "
                "an unsafe instruction pattern."
            )

        if "dangerous command" in result.reason.lower():

            return (
                "I can't process that request because it contains "
                "an unsupported or unsafe command."
            )

        if result.pii_detected:

            return (
                "I removed sensitive personal information from your "
                "request and processed the safe version."
            )

        return "I can't process that request."