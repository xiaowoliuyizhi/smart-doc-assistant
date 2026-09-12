"""
文档导入路由 —— POST /api/v1/document/import, /upload, GET /list
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import (
    DocListItem,
    DocListResponse,
    DocumentImportRequest,
    DocumentImportResponse,
)
from app.services.document_service import DocumentChunker
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/v1/document", tags=["文档管理"])


def _process_document(content: str, title: str, content_type: str, doc_id: str = "") -> dict:
    """公共处理流程：切片 → 向量化 → 存储"""
    if not doc_id:
        doc_id = uuid.uuid4().hex[:12]

    # 1. 切片
    chunker = DocumentChunker()
    chunks = chunker.split(content=content, content_type=content_type, doc_id=doc_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="切片结果为空，请检查文档内容")

    # 2. 向量化
    embedder = EmbeddingService()
    texts = [c.content for c in chunks]
    try:
        embeddings = embedder.embed_batch(texts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"向量化失败: {str(e)}")

    # 3. 存储
    store = VectorStore()
    chunk_dicts = [
        {"content": c.content, "chunk_index": c.chunk_index, "heading_path": c.heading_path}
        for c in chunks
    ]
    try:
        store.store_chunks(doc_id, chunk_dicts, embeddings)
        store.set_doc_meta(doc_id, title, len(chunks), len(content), content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"存储失败: {str(e)}")

    return {
        "doc_id": doc_id,
        "title": title,
        "chunk_count": len(chunks),
        "total_chars": len(content),
    }


@router.post(
    "/import",
    response_model=DocumentImportResponse,
    summary="导入文档（文本）",
    description="直接提交文本内容，系统将自动切片、向量化并存入知识库。",
)
async def import_document(req: DocumentImportRequest):
    if not req.content or not req.content.strip():
        raise HTTPException(status_code=400, detail="文档内容不能为空")
    result = _process_document(req.content, req.title or "", req.content_type, req.doc_id or "")
    return DocumentImportResponse(**result)


@router.post(
    "/upload",
    response_model=DocumentImportResponse,
    summary="上传文件导入",
    description="上传 .txt / .md / .docx 文件，自动识别格式、切片并向量化存储。",
)
async def upload_document(
    file: UploadFile = File(..., description="文档文件（支持 .txt / .md / .docx / .pdf）"),
    title: str = Form("", description="文档标题，不填则用文件名"),
):
    # 读取文件内容
    raw = await file.read()
    filename = file.filename or "未命名文档"

    # 判断文件类型
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        try:
            from io import BytesIO

            import fitz  # PyMuPDF
            doc = fitz.open(stream=raw, filetype="pdf")
            pages_text = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages_text.append(text.strip())
            doc.close()
            content = "\n\n".join(pages_text)
            content_type = "text"
            if not content.strip():
                raise HTTPException(status_code=400, detail="PDF 文件无法提取文字内容，可能是扫描件或图片型 PDF")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法解析 PDF 文件: {str(e)}")
    elif ext == "docx":
        # 解析 Word 文档
        try:
            from io import BytesIO

            from docx import Document
            doc = Document(BytesIO(raw))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            content = "\n\n".join(paragraphs)
            content_type = "text"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法解析 Word 文档: {str(e)}")
    elif ext == "md":
        content = raw.decode("utf-8", errors="replace")
        content_type = "markdown"
    elif ext == "txt" or ext == "":
        content = raw.decode("utf-8", errors="replace")
        content_type = "text"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: .{ext}（仅支持 .txt / .md / .docx / .pdf）",
        )

    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="文件内容为空或无法解析")

    doc_title = title.strip() if title.strip() else filename
    result = _process_document(content, doc_title, content_type)
    return DocumentImportResponse(**result)


@router.get(
    "/list",
    response_model=DocListResponse,
    summary="查看知识库文档列表",
    description="列出所有已导入的文档及其切片数量、字符数等摘要信息。",
)
async def list_documents():
    store = VectorStore()
    docs = store.list_docs()
    return DocListResponse(
        documents=[DocListItem(**d) for d in docs],
        total=len(docs),
    )


@router.delete(
    "/{doc_id}",
    summary="删除文档",
    description="从知识库中删除指定文档的所有切片及元信息。",
)
async def delete_document(doc_id: str):
    store = VectorStore()
    deleted = store.delete_doc(doc_id)
    return {"doc_id": doc_id, "deleted": deleted > 0}
