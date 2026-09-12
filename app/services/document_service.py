"""
文档切片服务 —— 将长文本拆分为适合检索的 Chunks。

切片策略：
  1. Markdown 文档：先按标题层级（#, ##, ###）切块，若某节超长则再用
     滑动窗口二次切分；保留标题路径作为上下文前缀。
  2. 纯文本文档：按段落（\\n\\n）初步分割，再用滑动窗口合并/切分到目标长度。
  3. 所有片段之间保留 overlap 字符的重叠，避免语义断裂。
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Iterator

from app.config import get_settings


@dataclass
class Chunk:
    """单个文本片段"""
    doc_id: str
    chunk_index: int
    content: str
    heading_path: str  # Markdown 标题路径，纯文本为空


class DocumentChunker:
    """文档切片器"""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = chunk_overlap or settings.chunk_overlap

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def split(
        self,
        content: str,
        content_type: str = "text",
        doc_id: str | None = None,
    ) -> list[Chunk]:
        """将文档内容切分为片段列表。"""
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        if content_type == "markdown":
            chunks = self._split_markdown(content, doc_id)
        else:
            chunks = self._split_plain_text(content, doc_id)
        # 重新编号
        for i, c in enumerate(chunks):
            c.chunk_index = i
        return chunks

    # ------------------------------------------------------------------
    # Markdown 切片
    # ------------------------------------------------------------------

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def _split_markdown(self, content: str, doc_id: str) -> list[Chunk]:
        """按 Markdown 标题层级切片，超长节二次滑窗切分。"""
        chunks: list[Chunk] = []
        # 找到所有标题位置
        headings = list(self._HEADING_RE.finditer(content))

        if not headings:
            # 没有标题，当作纯文本处理
            return self._split_plain_text(content, doc_id)

        # 前导内容（第一个标题之前）
        if headings[0].start() > 0:
            preface = content[: headings[0].start()].strip()
            if preface:
                chunks.extend(self._sliding_window(preface, doc_id, ""))

        # 遍历每个标题块
        heading_stack: list[tuple[int, str]] = []  # (level, title)
        for i, match in enumerate(headings):
            level = len(match.group(1))
            title = match.group(2).strip()

            # 维护标题栈
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            heading_path = " > ".join(t for _, t in heading_stack)

            # 当前节的内容范围
            section_start = match.start()
            section_end = (
                headings[i + 1].start() if i + 1 < len(headings) else len(content)
            )
            section_text = content[section_start:section_end].strip()

            # 若超长则滑窗切分
            if len(section_text) > self.chunk_size:
                chunks.extend(
                    self._sliding_window(section_text, doc_id, heading_path)
                )
            else:
                chunks.append(
                    Chunk(doc_id, 0, section_text, heading_path)
                )

        return chunks

    # ------------------------------------------------------------------
    # 纯文本切片
    # ------------------------------------------------------------------

    def _split_plain_text(self, content: str, doc_id: str) -> list[Chunk]:
        """纯文本：先按段落分割，再滑窗合并到目标长度。"""
        return self._sliding_window(content.strip(), doc_id, "")

    # ------------------------------------------------------------------
    # 滑动窗口
    # ------------------------------------------------------------------

    def _sliding_window(
        self, text: str, doc_id: str, heading_path: str
    ) -> list[Chunk]:
        """滑动窗口切分：按 chunk_size 切块，相邻块重叠 overlap 字符。

        切分时尽量在段落/句子边界处断开，避免硬截断。
        """
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                # 最后一块
                chunk_text = text[start:].strip()
                if chunk_text:
                    chunks.append(
                        Chunk(doc_id, 0, chunk_text, heading_path)
                    )
                break

            # 尝试在段落或句子边界处断开（向前回退）
            boundary = self._find_boundary(text, end)
            chunk_text = text[start:boundary].strip()
            if chunk_text:
                # 如果有标题路径，作为前缀附加
                final_text = (
                    f"[{heading_path}]\n{chunk_text}" if heading_path else chunk_text
                )
                chunks.append(Chunk(doc_id, 0, final_text, heading_path))

            # 下一块起点（回退 overlap）
            start = max(boundary - self.overlap, start + 1)

        return chunks

    @staticmethod
    def _find_boundary(text: str, pos: int) -> int:
        """在 pos 附近向前寻找最近的段落/句子边界。"""
        # 优先找段落分隔
        for pattern in ["\n\n", "\n", "。", ". ", "！", "？", "; ", "；"]:
            idx = text.rfind(pattern, pos - 100, pos)
            if idx != -1 and idx > pos - 100:
                return idx + len(pattern)
        return pos
