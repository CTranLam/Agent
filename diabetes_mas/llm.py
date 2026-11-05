"""LLM tuỳ chọn. Không có khoá thì hệ thống vẫn chạy bằng chế độ trích xuất."""
import os

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"


class LLM:
    def __init__(self, client=None, model: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.enabled = client is not None or bool(os.getenv("ANTHROPIC_API_KEY"))
        self.client = client or (anthropic.Anthropic() if self.enabled else None)
        self.last_error: str | None = None

    def complete(self, system: str, user: str) -> str | None:
        """Trả văn bản, hoặc None khi lỗi: lỗi không bao giờ được hiện ra như một câu trả lời."""
        if not self.enabled:
            return None
        self.last_error = None
        try:
            r = self.client.beta.messages.create(
                model=self.model, max_tokens=16000, system=system,
                output_config={"effort": "low"},  # chỉ diễn đạt lại, không cần suy luận sâu
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": user}])
        except anthropic.APIError as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return None
        if r.stop_reason == "refusal":
            self.last_error = "mô hình từ chối trả lời"
            return None
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        if not text:
            self.last_error = "không có phản hồi"
        return text or None
