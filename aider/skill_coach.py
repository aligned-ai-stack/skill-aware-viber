import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SkillAssessment:
    level: str
    confidence: str
    should_scaffold: bool
    blind_reliance_risk: bool
    rationale: tuple[str, ...]


NOVICE_PATTERNS = [
    r"\b(?:i am|i'm|im) (?:new|a beginner|beginner|novice)\b",
    r"\b(?:i don't|i do not|dont) understand\b",
    r"\bno idea\b",
    r"\bjust fix (?:it|this)\b",
    r"\bfix this\b",
    r"\bdo it for me\b",
    r"\bwhat do i do\b",
    r"\bhelp me debug\b",
]

LEARNING_PATTERNS = [
    r"\bteach me\b",
    r"\bhelp me understand\b",
    r"\bexplain (?:why|what)\b",
    r"\bwalk me through\b",
    r"\bso i can learn\b",
]

INTERMEDIATE_PATTERNS = [
    r"\bi tried\b",
    r"\bi suspect\b",
    r"\blikely cause\b",
    r"\bconfidence: (?:low|medium|high)\b",
    r"\bobserved issue\b",
    r"\bfailure target\b",
]

EXPERT_PATTERNS = [
    r"\broot cause\b",
    r"\bminimal repro\b",
    r"\bregression\b",
    r"\btrace through\b",
    r"\bnarrow it down\b",
]

VAGUE_DEBUG_PATTERNS = [
    r"\bfix bugs?\b",
    r"\bfix this\b",
    r"\bmake it work\b",
    r"\bdebug this\b",
]


def _matches(text: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text)]


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _extract_response_line(response: str, keywords: list[str]) -> str | None:
    for raw_line in response.splitlines():
        line = _clean_line(raw_line)
        if not line or line.startswith("@@") or line.startswith("+++") or line.startswith("---"):
            continue
        lower = line.lower()
        if any(keyword in lower for keyword in keywords):
            return line
    return None


def assess_user_skill(message: str) -> SkillAssessment:
    text = (message or "").lower()

    novice_hits = _matches(text, NOVICE_PATTERNS)
    learning_hits = _matches(text, LEARNING_PATTERNS)
    intermediate_hits = _matches(text, INTERMEDIATE_PATTERNS)
    expert_hits = _matches(text, EXPERT_PATTERNS)
    vague_hits = _matches(text, VAGUE_DEBUG_PATTERNS)

    blind_reliance_risk = bool(
        novice_hits
        or vague_hits
        or re.search(r"\b(?:blindly relies?|just make it work|whatever fixes it)\b", text)
    )

    if novice_hits:
        level = "novice"
        confidence = "medium" if intermediate_hits else "high"
    elif expert_hits:
        level = "expert"
        confidence = "medium"
    elif intermediate_hits:
        level = "intermediate"
        confidence = "medium"
    else:
        level = "intermediate"
        confidence = "low"

    should_scaffold = bool(learning_hits or vague_hits or blind_reliance_risk or level == "novice")

    rationale = tuple(
        [
            *[f"novice:{hit}" for hit in novice_hits],
            *[f"learning:{hit}" for hit in learning_hits],
            *[f"intermediate:{hit}" for hit in intermediate_hits],
            *[f"expert:{hit}" for hit in expert_hits],
            *[f"vague:{hit}" for hit in vague_hits],
        ]
    )

    return SkillAssessment(
        level=level,
        confidence=confidence,
        should_scaffold=should_scaffold,
        blind_reliance_risk=blind_reliance_risk,
        rationale=rationale,
    )


def build_skill_aware_prompt(message: str) -> str:
    assessment = assess_user_skill(message)

    if not assessment.should_scaffold:
        return message

    guidance = [
        "Use a teaching-oriented debugging style for this request.",
        f"Inferred skill level: {assessment.level} ({assessment.confidence} confidence).",
    ]

    if assessment.blind_reliance_risk:
        guidance.append("The user may be at risk of blind reliance or frustration.")

    guidance.extend(
        [
            "Before proposing a fix, briefly explain the likely failure mode in plain language.",
            "Give a short debugging path the user can follow to verify the cause.",
            "If you edit code, explain what each change addresses and what to watch for next.",
            "Keep jargon minimal and define it when needed.",
            "Aim to help the user learn, not just unblock them.",
        ]
    )

    return "\n".join(
        [
            "<skill_aware_debugging>",
            *guidance,
            "</skill_aware_debugging>",
            "",
            message,
        ]
    )


def format_skill_coach_summary(user_message: str, assistant_response: str, edited: bool = False) -> str | None:
    assessment = assess_user_skill(user_message)
    if not assessment.should_scaffold:
        return None

    request = _clean_line(user_message)
    diagnosis = _extract_response_line(
        assistant_response,
        ["because", "returns", "expects", "undefined", "none", "typeerror", "attributeerror", "syntax", "unpack", "error", "bug"],
    )
    if not diagnosis:
        diagnosis = "The model did not clearly isolate a root cause, so verify the exact failing line before trusting the edit."

    fix_line = _extract_response_line(
        assistant_response,
        ["change", "updated", "fixed", "replace", "for i, c in", "guard", "initialize"],
    )
    if edited and not fix_line:
        fix_line = "An edit was attempted. Inspect `/diff` or open the file to confirm the exact code change."
    elif not fix_line:
        fix_line = "No concrete fix was made visible yet. Ask for the smallest correct change on the failing line."

    verify = "Run the smallest repro again and confirm the observed error is gone and the expected output is correct."
    if "print(" in assistant_response:
        verify = "Run the script again and compare the printed output against the expected result for the sample input."

    lines = [
        "## Inferred Debug Brief",
        f"Request looks {assessment.level}-leaning ({assessment.confidence} confidence). Goal: solve the bug and make the reasoning visible.",
    ]

    if _matches(request.lower(), VAGUE_DEBUG_PATTERNS):
        lines.extend(
            [
                "",
                "## Better Debug Query",
                f"Fix the concrete bug in `{request.split()[-1]}` if that file name looks right, explain the exact failing line, and make the smallest correct change.",
            ]
        )

    lines.extend(
        [
            "",
            "## Diagnosis",
            diagnosis,
            "",
            "## Fix",
            fix_line,
            "",
            "## Verification",
            verify,
        ]
    )

    if assessment.blind_reliance_risk:
        lines.extend(
            [
                "",
                "## Why This Happened",
                "The original request was vague enough that the model could sound confident without pinning the bug to a concrete failing line.",
            ]
        )

    mismatch = _extract_response_line(assistant_response, ["edge cases", "off-by-one", "rest of the file remains unchanged", "seen[c] = i + 1", "print(longest)+print(longest)"])
    if mismatch:
        lines.extend(
            [
                "",
                "## Attempt Comparison",
                f"The model also said: {mismatch}",
            ]
        )

    return "\n".join(lines)
