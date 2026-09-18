import re
from dataclasses import dataclass


# ============================================================
# 1. GUARDRAILS
# ============================================================

class Guardrail:
    """Basic input and output safety checks."""

    def check_input(self, text: str) -> bool:
        if not text:
            return False

        # Prevent extremely large prompts
        if len(text) > 10000:
            return False

        return True

    def check_output(self, text: str) -> bool:
        if not text:
            return False

        # Prevent sensitive information in output
        blocked = [
            "api_key=",
            "password=",
            "secret=",
            "system prompt"
        ]

        for word in blocked:
            if word.lower() in text.lower():
                return False

        return True


# ============================================================
# 2. PROMPT INJECTION DEFENSE
# ============================================================

class PromptInjectionDefense:
    """Detect common prompt injection attacks."""

    def __init__(self):
        self.patterns = [
            r"ignore previous instructions",
            r"ignore all instructions",
            r"forget previous instructions",
            r"ignore the system prompt",
            r"reveal system prompt",
            r"show system prompt",
            r"disable safety",
            r"bypass safety",
            r"jailbreak",
            r"enter developer mode"
        ]

    def detect(self, text: str) -> bool:
        for pattern in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def check(self, text: str) -> bool:
        return not self.detect(text)


# ============================================================
# 3. PII PROTECTION
# ============================================================

class PIIProtector:
    """Detect and mask common PII."""

    def detect(self, text: str) -> list:
        pii = []

        # Email
        if re.search(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text
        ):
            pii.append("email")

        # Phone
        if re.search(r"\b\d{10}\b", text):
            pii.append("phone")

        # Credit card
        if re.search(r"\b(?:\d{4}[- ]?){3}\d{4}\b", text):
            pii.append("credit_card")

        return pii

    def mask(self, text: str) -> str:

        # Mask email
        text = re.sub(
            r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
            r"\1***@\2",
            text
        )

        # Mask phone
        text = re.sub(
            r"\b\d{6}(\d{4})\b",
            r"******\1",
            text
        )

        # Mask credit card
        text = re.sub(
            r"\b(?:\d{4}[- ]?){3}(\d{4})\b",
            r"****-****-****-\1",
            text
        )

        return text


# ============================================================
# 4. ACCESS CONTROL
# ============================================================

@dataclass
class User:
    user_id: str
    role: str


class AccessControl:
    """Simple RBAC authorization."""

    permissions = {
        "user": {
            "weather:read"
        },

        "manager": {
            "weather:read",
            "task:create"
        },

        "admin": {
            "weather:read",
            "task:create",
            "user:read",
            "user:write"
        }
    }

    def is_allowed(self, user: User, permission: str) -> bool:

        user_permissions = self.permissions.get(
            user.role,
            set()
        )

        return permission in user_permissions

    def can_use_tool(self, user: User, tool_name: str) -> bool:

        tool_permissions = {
            "get_weather": "weather:read",
            "create_task": "task:create",
            "get_user": "user:read"
        }

        required_permission = tool_permissions.get(tool_name)

        if not required_permission:
            return False

        return self.is_allowed(
            user,
            required_permission
        )


# ============================================================
# 5. HALLUCINATION MITIGATION / GROUNDING
# ============================================================

class GroundingValidator:
    """Make sure answers are based on trusted information."""

    def validate(
        self,
        answer: str,
        sources: list
    ) -> bool:

        # No source = cannot verify answer
        if not sources:
            return False

        # Answer must contain a source reference
        for source in sources:
            if source in answer:
                return True

        return False


# ============================================================
# COMPLETE AGENT SAFETY
# ============================================================

class AgentSafety:
    """Single safety layer used by the agent."""

    def __init__(self):

        self.guardrail = Guardrail()

        self.prompt_injection = PromptInjectionDefense()

        self.pii = PIIProtector()

        self.access = AccessControl()

        self.grounding = GroundingValidator()

    # --------------------------------------------------------
    # INPUT SAFETY
    # --------------------------------------------------------

    def validate_input(self, text: str) -> bool:

        # Guardrail
        if not self.guardrail.check_input(text):
            return False

        # Prompt injection
        if not self.prompt_injection.check(text):
            return False

        return True

    # --------------------------------------------------------
    # PII PROTECTION
    # --------------------------------------------------------

    def protect_pii(self, text: str) -> str:

        return self.pii.mask(text)

    # --------------------------------------------------------
    # TOOL SAFETY
    # --------------------------------------------------------

    def validate_tool(
        self,
        user: User,
        tool_name: str
    ) -> bool:

        return self.access.can_use_tool(
            user,
            tool_name
        )

    # --------------------------------------------------------
    # OUTPUT SAFETY
    # --------------------------------------------------------

    def validate_output(self, text: str) -> bool:

        return self.guardrail.check_output(text)

    # --------------------------------------------------------
    # GROUNDING
    # --------------------------------------------------------

    def validate_grounding(
        self,
        answer: str,
        sources: list
    ) -> bool:

        return self.grounding.validate(
            answer,
            sources
        )