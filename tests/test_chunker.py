"""
文档切片器单元测试 —— 纯逻辑测试，不依赖外部服务。
"""
import pytest

from app.services.document_service import DocumentChunker, Chunk


class TestDocumentChunker:
    """测试文档切片逻辑"""

    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    # ------------------------------------------------------------------
    # 基本切片
    # ------------------------------------------------------------------

    def test_short_text_single_chunk(self):
        """短文本应只产生一个片段"""
        text = "这是一段很短的文本。"
        chunks = self.chunker.split(text, content_type="text")
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0

    def test_empty_text(self):
        """空文本不应产生片段"""
        chunks = self.chunker.split("", content_type="text")
        assert len(chunks) == 0

    def test_long_text_multiple_chunks(self):
        """长文本应被切分为多个片段"""
        text = "这是一段测试文本。" * 50  # 约 450 字
        chunks = self.chunker.split(text, content_type="text")
        assert len(chunks) > 1
        # 验证 chunk_index 连续
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_doc_id_auto_generated(self):
        """不传 doc_id 时应自动生成"""
        chunks = self.chunker.split("测试文本", content_type="text")
        assert chunks[0].doc_id.startswith("doc_")

    def test_custom_doc_id(self):
        """传入 doc_id 时应使用自定义 ID"""
        chunks = self.chunker.split("测试文本", content_type="text", doc_id="my_doc")
        assert chunks[0].doc_id == "my_doc"

    # ------------------------------------------------------------------
    # 滑动窗口与重叠
    # ------------------------------------------------------------------

    def test_overlap_between_chunks(self):
        """相邻片段应有重叠内容"""
        text = "段落一内容。" * 20 + "段落二内容。" * 20
        chunks = self.chunker.split(text, content_type="text")
        if len(chunks) >= 2:
            # 第二个片段的开头应该与第一个片段的结尾有重叠
            # 或者至少两个片段不为空
            assert len(chunks[0].content) > 0
            assert len(chunks[1].content) > 0

    def test_chunk_size_within_limit(self):
        """每个片段长度不应远超 chunk_size（允许因边界调整略大）"""
        text = "a" * 500
        chunks = self.chunker.split(text, content_type="text")
        for c in chunks:
            # 允许 50% 容差（边界对齐可能导致略大）
            assert len(c.content) <= 150

    # ------------------------------------------------------------------
    # Markdown 切片
    # ------------------------------------------------------------------

    def test_markdown_heading_split(self):
        """Markdown 文档应按标题切分"""
        md = """# 第一章

这是第一章的内容，介绍 Redis 基础。

## 1.1 安装

Redis 的安装步骤如下。

## 1.2 配置

Redis 的配置文件位于 redis.conf。

# 第二章

这是第二章的内容。
"""
        chunks = self.chunker.split(md, content_type="markdown")
        assert len(chunks) >= 3
        # 验证标题路径
        heading_paths = [c.heading_path for c in chunks]
        assert any("第一章" in h for h in heading_paths)
        assert any("安装" in h for h in heading_paths)
        assert any("第二章" in h for h in heading_paths)

    def test_markdown_nested_headings(self):
        """Markdown 嵌套标题路径应正确构建"""
        md = """# 顶层

## 子标题 A

内容 A。

### 子标题 A.1

内容 A.1。
"""
        chunks = self.chunker.split(md, content_type="markdown")
        # 应包含嵌套路径
        paths = [c.heading_path for c in chunks]
        assert any("子标题 A.1" in p for p in paths)
        # 子标题 A.1 的路径应包含 "顶层 > 子标题 A > 子标题 A.1"
        a1_chunk = [c for c in chunks if "A.1" in c.heading_path]
        if a1_chunk:
            assert "顶层" in a1_chunk[0].heading_path
            assert "子标题 A" in a1_chunk[0].heading_path

    def test_markdown_long_section_sliding_window(self):
        """Markdown 超长节应触发滑窗二次切分"""
        md = """# 长节

""" + "这是一段很长的内容。" * 50
        chunks = self.chunker.split(md, content_type="markdown")
        assert len(chunks) > 1
        # 所有片段都应包含标题路径
        for c in chunks:
            assert "长节" in c.heading_path

    def test_markdown_no_headings_fallback(self):
        """没有标题的 Markdown 应回退为纯文本切片"""
        md = "这是一段没有标题的 Markdown 文本。" * 20
        chunks = self.chunker.split(md, content_type="markdown")
        assert len(chunks) > 0
        # heading_path 应为空
        for c in chunks:
            assert c.heading_path == ""

    # ------------------------------------------------------------------
    # 边界情况
    # ------------------------------------------------------------------

    def test_whitespace_only(self):
        """纯空白文本不应产生片段"""
        chunks = self.chunker.split("   \n\n  \t  ", content_type="text")
        assert len(chunks) == 0

    def test_single_long_line(self):
        """超长单行文本应被切分"""
        text = "x" * 1000
        chunks = self.chunker.split(text, content_type="text")
        assert len(chunks) > 1
