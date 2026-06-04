from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Pattern

from ..models import Category, DetectorInfo, Finding, Severity
from .base import Detector, detector_status, finding_context_fields


@dataclass(frozen=True)
class RegexRule:
    """
    A single regex-based detection rule.

    Rules are intentionally small and explicit. The detector does not decide
    final risk by itself; it only emits normalized findings.
    """

    rule_id: str
    category: Category
    severity: Severity
    pattern: Pattern[str]
    message: str
    subtype: str | None = None
    confidence: float = 0.8


def compile_rule(
    *,
    rule_id: str,
    category: Category,
    severity: Severity,
    pattern: str,
    message: str,
    subtype: str | None = None,
    confidence: float = 0.8,
    flags: int = re.IGNORECASE | re.MULTILINE,
) -> RegexRule:
    """
    Compile a regex rule and return a RegexRule instance.
    """

    return RegexRule(
        rule_id=rule_id,
        category=category,
        severity=severity,
        pattern=re.compile(pattern, flags),
        message=message,
        subtype=subtype,
        confidence=confidence,
    )


DEFAULT_REGEX_RULES: list[RegexRule] = [
    # -------------------------------------------------------------------------
    # Prompt injection
    # -------------------------------------------------------------------------
    compile_rule(
        rule_id="prompt.role_boundary_marker",
        category=Category.PROMPT_INJECTION,
        subtype="ROLE_BOUNDARY_CONFUSION",
        severity=Severity.REVIEW,
        pattern=(
            r"(?:"
            r"<\|\s*(?:system|developer|assistant|tool|user)\s*\|>"
            r"|"
            r"#{2,}\s*(?:system|developer|assistant|tool|user)\s*:"
            r"|"
            r"\bBEGIN\s+(?:SYSTEM|DEVELOPER|ASSISTANT|TOOL|USER)\s+MESSAGE\b"
            r"|"
            r"\bEND\s+(?:SYSTEM|DEVELOPER|ASSISTANT|TOOL|USER)\s+MESSAGE\b"
            r")"
        ),
        message="Possible prompt injection role-boundary marker.",
        confidence=0.8,
    ),
    compile_rule(
        rule_id="prompt.system_prompt_disclosure",
        category=Category.PROMPT_INJECTION,
        subtype="SYSTEM_PROMPT_DISCLOSURE",
        severity=Severity.REVIEW,
        pattern=(
            r"(?:"
            r"(?:reveal|disclose|show|print|return|dump|exfiltrate)\s+"
            r"(?:the\s+)?"
            r"(?:system\s+prompt|developer\s+message|hidden\s+instructions|"
            r"tool\s+instructions|previous\s+messages|confidential\s+prompt)"
            r"|"
            r"(?:system\s+prompt|developer\s+message|hidden\s+instructions|"
            r"tool\s+instructions|previous\s+messages|confidential\s+prompt)\s+"
            r"(?:must\s+be\s+)?(?:revealed|disclosed|shown|printed|returned|dumped)"
            r")"
        ),
        message="Possible prompt injection attempt to disclose hidden instructions.",
        confidence=0.9,
    ),
    compile_rule(
        rule_id="prompt.developer_message_override",
        category=Category.PROMPT_INJECTION,
        subtype="DEVELOPER_MESSAGE_OVERRIDE",
        severity=Severity.REVIEW,
        pattern=(
            r"(?:"
            r"(?:override|replace|ignore|bypass)\s+"
            r"(?:the\s+)?(?:system|developer|assistant|tool)\s+"
            r"(?:message|instructions|prompt|policy)"
            r"|"
            r"(?:act|respond|behave)\s+as\s+"
            r"(?:the\s+)?(?:system|developer|admin|root)"
            r"|"
            r"developer\s+mode"
            r")"
        ),
        message="Possible prompt injection attempt to override role or developer instructions.",
        confidence=0.85,
    ),
    compile_rule(
        rule_id="prompt.ignore_previous_instructions",
        category=Category.PROMPT_INJECTION,
        subtype="IGNORE_PREVIOUS_INSTRUCTIONS",
        severity=Severity.BLOCK,
        pattern=r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?\b",
        message="Possible prompt injection instruction override.",
        confidence=0.9,
    ),
    compile_rule(
        rule_id="prompt.system_prompt_leak",
        category=Category.PROMPT_INJECTION,
        subtype="SYSTEM_PROMPT_LEAK",
        severity=Severity.BLOCK,
        pattern=r"\b(?:reveal|show|print|dump|exfiltrate)\s+(?:the\s+)?(?:system|developer)\s+prompt\b",
        message="Possible attempt to reveal hidden system or developer instructions.",
        confidence=0.9,
    ),
    compile_rule(
        rule_id="prompt.role_override",
        category=Category.PROMPT_INJECTION,
        subtype="ROLE_OVERRIDE",
        severity=Severity.REVIEW,
        pattern=r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:a\s+)?(?:root|admin|developer|system|jailbroken)\b",
        message="Possible role override prompt injection pattern.",
        confidence=0.75,
    ),
    compile_rule(
        rule_id="prompt.jailbreak_mode",
        category=Category.PROMPT_INJECTION,
        subtype="JAILBREAK_MODE",
        severity=Severity.REVIEW,
        pattern=r"\b(?:dan\s+mode|developer\s+mode|jailbreak|do\s+anything\s+now)\b",
        message="Possible jailbreak-style prompt injection pattern.",
        confidence=0.75,
    ),
    compile_rule(
        rule_id="prompt.policy_bypass",
        category=Category.PROMPT_INJECTION,
        subtype="POLICY_BYPASS",
        severity=Severity.REVIEW,
        pattern=r"\b(?:bypass|disable|ignore|override)\s+(?:safety|policy|guardrails?|filters?|restrictions?)\b",
        message="Possible safety policy bypass instruction.",
        confidence=0.8,
    ),

    # -------------------------------------------------------------------------
    # XSS
    # -------------------------------------------------------------------------
    compile_rule(
        rule_id="xss.script_tag",
        category=Category.XSS,
        subtype="SCRIPT_TAG",
        severity=Severity.BLOCK,
        pattern=r"<\s*script\b[^>]*>",
        message="Possible XSS script tag.",
        confidence=0.95,
    ),
    compile_rule(
        rule_id="xss.javascript_uri",
        category=Category.XSS,
        subtype="JAVASCRIPT_URI",
        severity=Severity.BLOCK,
        pattern=r"\bjavascript\s*:",
        message="Possible XSS javascript URI.",
        confidence=0.9,
    ),
    compile_rule(
        rule_id="xss.event_handler",
        category=Category.XSS,
        subtype="HTML_EVENT_HANDLER",
        severity=Severity.REVIEW,
        pattern=r"\bon(?:load|error|click|mouseover|focus|submit|mouseenter|mouseleave)\s*=",
        message="Possible XSS HTML event handler.",
        confidence=0.85,
    ),
    compile_rule(
        rule_id="xss.svg_onload",
        category=Category.XSS,
        subtype="SVG_ONLOAD",
        severity=Severity.BLOCK,
        pattern=r"<\s*svg\b[^>]*\bonload\s*=",
        message="Possible SVG-based XSS payload.",
        confidence=0.9,
    ),
    compile_rule(
        rule_id="xss.iframe_srcdoc",
        category=Category.XSS,
        subtype="IFRAME_SRCDOC",
        severity=Severity.REVIEW,
        pattern=r"<\s*iframe\b[^>]*\bsrcdoc\s*=",
        message="Possible iframe srcdoc XSS payload.",
        confidence=0.8,
    ),

    # -------------------------------------------------------------------------
    # SQL injection
    # -------------------------------------------------------------------------
    compile_rule(
        rule_id="sqli.union_select",
        category=Category.SQLI,
        subtype="UNION_SELECT",
        severity=Severity.BLOCK,
        pattern=r"\bunion\s+(?:all\s+)?select\b",
        message="Possible SQL injection UNION SELECT pattern.",
        confidence=0.9,
    ),
    compile_rule(
        rule_id="sqli.boolean_tautology",
        category=Category.SQLI,
        subtype="BOOLEAN_TAUTOLOGY",
        severity=Severity.REVIEW,
        pattern=r"(?:'|\")?\s*(?:or|and)\s+(?:'|\")?\d+(?:'|\")?\s*=\s*(?:'|\")?\d+",
        message="Possible SQL injection boolean tautology.",
        confidence=0.75,
    ),
    compile_rule(
        rule_id="sqli.comment_sequence",
        category=Category.SQLI,
        subtype="SQL_COMMENT_SEQUENCE",
        severity=Severity.REVIEW,
        pattern=r"(?:--|#|/\*)\s*$",
        message="Possible SQL injection comment sequence.",
        confidence=0.65,
    ),
    compile_rule(
        rule_id="sqli.stacked_query",
        category=Category.SQLI,
        subtype="STACKED_QUERY",
        severity=Severity.BLOCK,
        pattern=r";\s*(?:drop|delete|insert|update|alter|create|truncate)\s+\w+",
        message="Possible SQL injection stacked query.",
        confidence=0.85,
    ),
    compile_rule(
        rule_id="sqli.time_based",
        category=Category.SQLI,
        subtype="TIME_BASED",
        severity=Severity.REVIEW,
        pattern=r"\b(?:sleep|benchmark|pg_sleep|waitfor\s+delay)\s*\(",
        message="Possible time-based SQL injection pattern.",
        confidence=0.8,
    ),

    # -------------------------------------------------------------------------
    # Command injection
    # -------------------------------------------------------------------------
    compile_rule(
        rule_id="cmd.shell_separator",
        category=Category.COMMAND_INJECTION,
        subtype="SHELL_SEPARATOR",
        severity=Severity.REVIEW,
        pattern=r"(?:;|\|\||&&)\s*(?:cat|curl|wget|bash|sh|powershell|cmd|whoami|id|nc|netcat)\b",
        message="Possible command injection shell separator.",
        confidence=0.8,
    ),
    compile_rule(
        rule_id="cmd.command_substitution",
        category=Category.COMMAND_INJECTION,
        subtype="COMMAND_SUBSTITUTION",
        severity=Severity.REVIEW,
        pattern=(
            r"(?:"
            r"`\s*(?:cat|curl|wget|bash|sh|powershell|pwsh|cmd|whoami|id|nc|netcat|python|python3|perl|ruby|php|node|npx|npm|pip|pip3|chmod|chown|rm|del|copy|cp|mv|tar|zip|unzip|scp|ssh)\b[^`]*`"
            r"|"
            r"\$\(\s*(?:cat|curl|wget|bash|sh|powershell|pwsh|cmd|whoami|id|nc|netcat|python|python3|perl|ruby|php|node|npx|npm|pip|pip3|chmod|chown|rm|del|copy|cp|mv|tar|zip|unzip|scp|ssh)\b[^)]*\)"
            r")"
        ),
        message="Possible shell command substitution.",
        confidence=0.8,
    ),
    compile_rule(
        rule_id="cmd.powershell_encoded",
        category=Category.COMMAND_INJECTION,
        subtype="POWERSHELL_ENCODED_COMMAND",
        severity=Severity.BLOCK,
        pattern=r"\bpowershell(?:\.exe)?\b.*\s-(?:enc|encodedcommand)\b",
        message="Possible encoded PowerShell command.",
        confidence=0.9,
    ),

    # -------------------------------------------------------------------------
    # Path traversal
    # -------------------------------------------------------------------------
    compile_rule(
        rule_id="path.traversal_dotdot",
        category=Category.PATH_TRAVERSAL,
        subtype="DOT_DOT_SLASH",
        severity=Severity.REVIEW,
        pattern=r"(?:\.\./|\.\.\\){2,}",
        message="Possible path traversal sequence.",
        confidence=0.85,
    ),
    compile_rule(
        rule_id="path.traversal_encoded",
        category=Category.PATH_TRAVERSAL,
        subtype="ENCODED_DOT_DOT_SLASH",
        severity=Severity.REVIEW,
        pattern=r"(?:%2e%2e%2f|%2e%2e/|%252e%252e%252f|%c0%ae)",
        message="Possible encoded path traversal sequence.",
        confidence=0.85,
    ),
    compile_rule(
        rule_id="path.sensitive_unix_file",
        category=Category.PATH_TRAVERSAL,
        subtype="SENSITIVE_UNIX_FILE",
        severity=Severity.REVIEW,
        pattern=r"/etc/(?:passwd|shadow|hosts)\b",
        message="Possible sensitive Unix file path reference.",
        confidence=0.8,
    ),
    compile_rule(
        rule_id="path.sensitive_windows_file",
        category=Category.PATH_TRAVERSAL,
        subtype="SENSITIVE_WINDOWS_FILE",
        severity=Severity.REVIEW,
        pattern=r"\bC:\\Windows\\(?:System32|win\.ini)",
        message="Possible sensitive Windows file path reference.",
        confidence=0.8,
    ),

    # -------------------------------------------------------------------------
    # Template injection
    # -------------------------------------------------------------------------
    compile_rule(
        rule_id="template.jinja_arithmetic",
        category=Category.TEMPLATE_INJECTION,
        subtype="JINJA_ARITHMETIC_EXPRESSION",
        severity=Severity.REVIEW,
        pattern=r"\{\{\s*\d+\s*[*+\-/]\s*\d+\s*\}\}",
        message="Possible template arithmetic expression injection.",
        confidence=0.75,
    ),
    compile_rule(
        rule_id="template.jndi_lookup",
        category=Category.TEMPLATE_INJECTION,
        subtype="JNDI_LOOKUP",
        severity=Severity.BLOCK,
        pattern=(
            r"\$\{\s*jndi\s*:\s*"
            r"(?:ldap|ldaps|rmi|dns|iiop|corba|nis|nds|http|https)\s*://"
        ),
        message="Possible JNDI lookup template injection payload.",
        confidence=0.98,
    ),
    compile_rule(
        rule_id="template.java_runtime_exec",
        category=Category.TEMPLATE_INJECTION,
        subtype="JAVA_RUNTIME_EXEC",
        severity=Severity.BLOCK,
        pattern=(
            r"(?:"
            r"\$\{\s*T\s*\(\s*java\.lang\.Runtime\s*\)"
            r"\s*\.getRuntime\s*\(\s*\)\s*\.exec\s*\("
            r"|"
            r"java\.lang\.Runtime\s*\.getRuntime\s*\(\s*\)\s*\.exec\s*\("
            r"|"
            r"Runtime\s*\.getRuntime\s*\(\s*\)\s*\.exec\s*\("
            r")"
        ),
        message="Possible Java runtime execution template payload.",
        confidence=0.98,
    ),
    compile_rule(
        rule_id="template.jsp_runtime_exec",
        category=Category.TEMPLATE_INJECTION,
        subtype="JSP_RUNTIME_EXEC",
        severity=Severity.BLOCK,
        pattern=(
            r"<%\s*=?\s*"
            r".{0,120}?"
            r"(?:Runtime\s*\.getRuntime\s*\(\s*\)\s*\.exec|"
            r"java\.lang\.Runtime\s*\.getRuntime\s*\(\s*\)\s*\.exec)"
        ),
        message="Possible JSP runtime execution payload.",
        confidence=0.98,
    ),
    compile_rule(
        rule_id="template.expression_runtime_exec",
        category=Category.TEMPLATE_INJECTION,
        subtype="EXPRESSION_RUNTIME_EXEC",
        severity=Severity.BLOCK,
        pattern=(
            r"(?:"
            r"\$\{\s*"
            r".{0,80}?"
            r"(?:exec\s*\(|popen\s*\(|ProcessBuilder\s*\(|getRuntime\s*\(\s*\))"
            r".{0,120}?"
            r"\}"
            r")"
        ),
        message="Possible expression-language runtime execution payload.",
        confidence=0.9,
    ),
    compile_rule(
        rule_id="template.jinja_function_call",
        category=Category.TEMPLATE_INJECTION,
        subtype="JINJA_FUNCTION_CALL",
        severity=Severity.REVIEW,
        pattern=r"\{\{\s*[a-zA-Z_][\w.]{0,80}\s*\([^{}]{0,200}\)\s*\}\}",
        message="Possible template function call injection.",
        confidence=0.75,
    ),

    compile_rule(
        rule_id="template.jinja_code_execution",
        category=Category.TEMPLATE_INJECTION,
        subtype="JINJA_CODE_EXECUTION",
        severity=Severity.BLOCK,
        pattern=(
            r"\{\{[^{}]{0,500}"
            r"(?:popen|subprocess|os\.|eval\s*\(|exec\s*\(|__import__\s*\()"
            r"[^{}]{0,500}\}\}"
        ),
        message="Possible dangerous Jinja/SSTI code execution pattern.",
        confidence=0.95,
    ),
    # -------------------------------------------------------------------------
    # LDAP injection
    # -------------------------------------------------------------------------
    compile_rule(
        rule_id="ldap.wildcard_filter",
        category=Category.LDAP_INJECTION,
        subtype="LDAP_WILDCARD_FILTER",
        severity=Severity.REVIEW,
        pattern=r"\(\s*\|\s*\([^()=]{1,64}=\*\)\s*\)",
        message="Possible LDAP wildcard filter injection.",
        confidence=0.75,
    ),
    compile_rule(
        rule_id="ldap.filter_breakout",
        category=Category.LDAP_INJECTION,
        subtype="LDAP_FILTER_BREAKOUT",
        severity=Severity.REVIEW,
        pattern=r"\)\s*\(\s*\|",
        message="Possible LDAP filter breakout sequence.",
        confidence=0.65,
    ),
]


class RegexDetector(Detector):
    """
    Fast regex-based detector for common security payloads.

    This detector is intentionally lightweight and dependency-free. It is useful
    as a first-pass scanner and as a fallback when specialized engines are not
    available.
    """

    name = "regex"

    def __init__(self, rules: list[RegexRule] | None = None) -> None:
        self.rules = rules or DEFAULT_REGEX_RULES

    def is_available(self) -> bool:
        return True

    def describe(self) -> DetectorInfo:
        return detector_status(
            name=self.name,
            available=True,
            metadata={
                "rule_count": len(self.rules),
                "categories": sorted({rule.category.value for rule in self.rules}),
            },
        )

    def scan_text(self, text: str, context: dict[str, Any] | None = None) -> list[Finding]:
        findings: list[Finding] = []
        context_fields = finding_context_fields(context)

        if not text:
            return findings

        for rule in self.rules:
            if not rule.pattern.search(text):
                continue

            findings.append(
                Finding(
                    category=rule.category,
                    subtype=rule.subtype,
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    message=rule.message,
                    detector=self.name,
                    confidence=rule.confidence,
                    **context_fields,
                )
            )

        return findings