"""One model-visible rendering for planning, recording and admission. No inference."""
from .reuse import legacy


def validate_messages(messages, *, inspect_serialized=False):
    if not isinstance(messages, list) or not messages:
        raise legacy.AnalysisError("Input messages must be an explicit nonempty list")
    normalized = []
    for message in messages:
        allowed = {"role", "content", "id"} if inspect_serialized else {"role", "content"}
        if (not isinstance(message, dict) or not {"role", "content"} <= set(message)
                or not set(message) <= allowed
                or message["role"] not in ("system", "user")
                or not isinstance(message["content"], str) or not message["content"].strip()):
            raise legacy.AnalysisError("Unsupported message role, content or fields")
        if "id" in message and (not isinstance(message["id"], str) or not message["id"]):
            raise legacy.AnalysisError("Invalid Inspect-generated message ID")
        # Inspect's random bookkeeping ID is not sent as model-visible content.
        # All other additional fields remain forbidden, including tool metadata.
        normalized.append({"role": message["role"], "content": message["content"]})
    return normalized


def render_messages(case, condition, spec):
    """Preserve the historical two-message input and system-appended prompt."""
    if condition not in spec["conditions"]:
        raise legacy.AnalysisError("Undeclared condition")
    scoring = legacy.scorer()
    scoring.validate_case(case)
    system, user = scoring.render_prompt(case)
    if (case.get("system_text"), case.get("user_text")) != (system, user):
        raise legacy.AnalysisError("Stored case messages differ from the canonical renderer")
    if condition == "prompt":
        suffix = spec.get("prompt_text")
        if not isinstance(suffix, str) or not suffix.strip():
            raise legacy.AnalysisError("The fixed prompt text is missing")
        system += "\n" + suffix
    return validate_messages([{"role": "system", "content": system}, {"role": "user", "content": user}])
