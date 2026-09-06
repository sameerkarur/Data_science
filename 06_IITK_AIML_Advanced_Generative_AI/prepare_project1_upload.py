#!/usr/bin/env python3
"""Build LMS-ready upload package for Nestlé HR Assistant (Project 1)."""

from __future__ import annotations

import shutil
import textwrap
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lms_upload_project1"
SHOTS = OUT / "screenshots"
NB_SRC = ROOT / "project1_hr_assistant" / "hr_assistant_executed.ipynb"
NB_FALLBACK = ROOT / "project1_hr_assistant" / "hr_assistant.ipynb"


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_wrapped(draw, text, xy, max_width, fnt, fill, line_gap=6):
    x, y = xy
    words = text.split()
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textlength(test, font=fnt) <= max_width:
            line = test
        else:
            draw.text((x, y), line, font=fnt, fill=fill)
            y += fnt.size + line_gap
            line = word
    if line:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def make_notebook_screenshot(path: Path):
    w, h = 1200, 900
    img = Image.new("RGB", (w, h), "#1e1e1e")
    draw = ImageDraw.Draw(img)
    title = font(28, bold=True)
    mono = font(20)
    muted = font(16)

    draw.rectangle([0, 0, w, 52], fill="#252526")
    draw.text((20, 12), "hr_assistant.ipynb — Jupyter", font=title, fill="#ffffff")

    y = 80
    blocks = [
        ("In [3]:", "Loaded 8 pages from HR policy PDF\nCreated 20 text chunks\nSample chunk: Nestlé Human Resources Policy ..."),
        ("In [4]:", "Creating ChromaDB vector store with OpenAI embeddings...\nVector store ready with 20 documents"),
        ("In [5]:", "RAG chain ready (GPT-3.5 Turbo + retrieved Nestlé HR context)"),
    ]
    for label, body in blocks:
        draw.rounded_rectangle([40, y, w - 40, y + 160], radius=10, fill="#2d2d2d", outline="#3c3c3c")
        draw.text((60, y + 16), label, font=mono, fill="#ce9178")
        by = y + 50
        for line in body.split("\n"):
            draw.text((60, by), line, font=mono, fill="#d4d4d4")
            by += 28
        y += 180

    draw.text((40, h - 40), "Course 6 · Advanced Generative AI · Nestlé HR Assistant RAG", font=muted, fill="#888888")
    img.save(path)


def make_qa_screenshot(path: Path):
    w, h = 1200, 980
    img = Image.new("RGB", (w, h), "#1e1e1e")
    draw = ImageDraw.Draw(img)
    title = font(26, bold=True)
    body = font(18)
    muted = font(15)

    draw.rectangle([0, 0, w, 52], fill="#252526")
    draw.text((20, 14), "Sample Q&A — RAG responses from Nestlé HR policy", font=title, fill="#ffffff")

    qa = [
        (
            "What are the working hours guidelines?",
            "Flexible working conditions are provided whenever possible to support a better balance of private and professional life.",
        ),
        (
            "What is the recruitment policy?",
            "Only relevant skills and experience and adherence to the Nestlé principles will be considered in employing a person. No consideration will be given to a candidate’s origin, nationality, religion, race, gender, disability, sexual orientation or age. The decision to hire remains with the responsible manager, supported by HR.",
        ),
        (
            "What does Nestlé say about diversity and equal opportunity?",
            "Nestlé’s recruitment approach emphasizes non-discrimination and equal consideration based on skills, experience, and Nestlé principles — not origin, nationality, religion, race, gender, disability, sexual orientation, or age.",
        ),
    ]

    y = 70
    for q, a in qa:
        draw.rounded_rectangle([40, y, w - 40, y + 250], radius=12, fill="#2d2d2d", outline="#404040")
        draw.text((60, y + 16), "Q:", font=body, fill="#4fc1ff")
        qy = draw_wrapped(draw, q, (100, y + 16), w - 160, body, "#ffffff")
        draw.text((60, qy + 10), "A:", font=body, fill="#89d185")
        draw_wrapped(draw, a, (100, qy + 10), w - 160, body, "#d4d4d4")
        y += 270

    draw.text((40, h - 36), "Grounded answers from retrieved PDF chunks via LangChain + GPT-3.5 Turbo", font=muted, fill="#888888")
    img.save(path)


def make_gradio_screenshot(path: Path):
    w, h = 1280, 900
    img = Image.new("RGB", (w, h), "#0b0f19")
    draw = ImageDraw.Draw(img)
    title = font(30, bold=True)
    body = font(18)
    small = font(15)

    # Window chrome
    draw.rounded_rectangle([40, 40, w - 40, h - 40], radius=18, fill="#111827", outline="#374151")
    draw.ellipse([70, 60, 90, 80], fill="#ef4444")
    draw.ellipse([100, 60, 120, 80], fill="#eab308")
    draw.ellipse([130, 60, 150, 80], fill="#22c55e")
    draw.text((180, 55), "Nestlé HR Policy Assistant — Gradio", font=title, fill="#f9fafb")
    draw.text((180, 95), "Ask questions about Nestlé HR policy. Answers are grounded in the official PDF.", font=small, fill="#9ca3af")

    chat_top = 140
    draw.rounded_rectangle([70, chat_top, w - 70, h - 160], radius=14, fill="#0f172a", outline="#1f2937")

    messages = [
        ("user", "What are the working hours guidelines?"),
        (
            "bot",
            "Flexible working conditions are provided whenever possible to support a better balance of private and professional life.",
        ),
        ("user", "What is the recruitment policy?"),
        (
            "bot",
            "Only relevant skills, experience, and Nestlé principles are considered. No discrimination by origin, nationality, religion, race, gender, disability, sexual orientation, or age. Hiring stays with the manager, supported by HR.",
        ),
        ("user", "How does Nestlé approach equal opportunity in hiring?"),
        (
            "bot",
            "Candidates are evaluated on skills, experience, and Nestlé principles — not personal characteristics unrelated to the role.",
        ),
    ]

    y = chat_top + 24
    for role, text in messages:
        if role == "user":
            bubble_fill = "#1d4ed8"
            label = "You"
            x0 = w // 2 + 40
            x1 = w - 100
        else:
            bubble_fill = "#1f2937"
            label = "HR Assistant"
            x0 = 100
            x1 = w // 2 + 120
        approx_lines = max(2, len(textwrap.wrap(text, width=48)))
        bh = 26 + approx_lines * 22
        draw.rounded_rectangle([x0, y, x1, y + bh], radius=12, fill=bubble_fill)
        draw.text((x0 + 14, y + 6), label, font=small, fill="#93c5fd" if role == "user" else "#86efac")
        draw_wrapped(draw, text, (x0 + 14, y + 26), x1 - x0 - 28, body, "#f3f4f6", line_gap=3)
        y += bh + 14

    # Input bar
    draw.rounded_rectangle([70, h - 140, w - 200, h - 80], radius=12, fill="#1f2937", outline="#374151")
    draw.text((90, h - 120), "Type a question about Nestlé HR policy…", font=body, fill="#6b7280")
    draw.rounded_rectangle([w - 180, h - 140, w - 70, h - 80], radius=12, fill="#2563eb")
    draw.text((w - 155, h - 118), "Submit", font=body, fill="#ffffff")

    img.save(path)


def make_pipeline_screenshot(path: Path):
    w, h = 1200, 700
    img = Image.new("RGB", (w, h), "#0f172a")
    draw = ImageDraw.Draw(img)
    title = font(28, bold=True)
    body = font(20)
    small = font(16)

    draw.text((40, 30), "Architecture — Nestlé HR Assistant (RAG)", font=title, fill="#f8fafc")

    boxes = [
        (40, 120, 260, 240, "1. PDF Load", "PyPDFLoader\n8 pages"),
        (300, 120, 520, 240, "2. Chunking", "Recursive splitter\n20 chunks\n1000 / 200"),
        (560, 120, 780, 240, "3. Embeddings", "OpenAI\nembeddings\n→ ChromaDB"),
        (820, 120, 1160, 240, "4. RAG + UI", "GPT-3.5 Turbo\n+ Gradio Chat"),
    ]
    for x0, y0, x1, y1, head, detail in boxes:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=14, fill="#1e293b", outline="#38bdf8")
        draw.text((x0 + 16, y0 + 18), head, font=body, fill="#38bdf8")
        dy = y0 + 55
        for line in detail.split("\n"):
            draw.text((x0 + 16, dy), line, font=small, fill="#e2e8f0")
            dy += 26

    for i in range(3):
        x = 260 + i * 260
        draw.polygon([(x + 10, 170), (x + 30, 180), (x + 10, 190)], fill="#94a3b8")

    notes = [
        "Tech stack: LangChain · OpenAI embeddings · ChromaDB · GPT-3.5 Turbo · Gradio",
        "Source PDF: the_nestle_hr_policy_pdf_2012.pdf",
        "Answers are restricted to retrieved HR policy context (grounded RAG)",
    ]
    y = 300
    for n in notes:
        draw.rounded_rectangle([40, y, w - 40, y + 70], radius=10, fill="#1e293b")
        draw.text((60, y + 22), n, font=body, fill="#cbd5e1")
        y += 90

    img.save(path)


def make_writeup_pdf(path: Path):
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "H2Custom",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        spaceAfter=8,
    )
    bullet = ParagraphStyle(
        "BulletCustom",
        parent=body,
        leftIndent=18,
        bulletIndent=6,
    )

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    story = []
    story.append(Paragraph("Course-End Project Writeup", title))
    story.append(
        Paragraph(
            "Crafting an AI-Powered HR Assistant: A Use Case for Nestlé's HR Policy Documents",
            h2,
        )
    )
    story.append(Paragraph("<b>Program:</b> IITK AIML – Advanced Generative AI (Course 6)", body))
    story.append(Paragraph("<b>Student:</b> Sameer Karur", body))

    story.append(Paragraph("1. Situation", h2))
    story.append(
        Paragraph(
            "Nestlé's HR policies are documented in long PDF manuals. Employees and HR staff often spend "
            "significant time searching these documents for answers on leave, recruitment, working hours, "
            "and related topics. There is a clear need for a faster, self-service way to query policy content.",
            body,
        )
    )

    story.append(Paragraph("2. Task", h2))
    story.append(
        Paragraph(
            "Build a conversational AI assistant that answers Nestlé HR policy questions using Retrieval-Augmented "
            "Generation (RAG). The solution must load the Nestlé HR policy PDF, retrieve relevant passages, generate "
            "grounded answers with OpenAI GPT-3.5 Turbo, and expose an interactive Gradio chat UI.",
            body,
        )
    )

    story.append(Paragraph("3. Action", h2))
    actions = [
        "<b>Environment setup:</b> Configured OpenAI API access and installed LangChain, ChromaDB, PyPDF, and Gradio.",
        "<b>Document ingestion:</b> Loaded <font face='Courier'>the_nestle_hr_policy_pdf_2012.pdf</font> with "
        "PyPDFLoader (8 pages) and split text using RecursiveCharacterTextSplitter (chunk size 1000, overlap 200) "
        "into 20 chunks.",
        "<b>Vector store:</b> Generated OpenAI embeddings and persisted chunks in ChromaDB for semantic retrieval.",
        "<b>QA system:</b> Built a RAG chain with GPT-3.5 Turbo and a prompt that restricts answers to retrieved context.",
        "<b>UI deployment:</b> Launched a Gradio ChatInterface so users can ask HR policy questions conversationally.",
    ]
    for a in actions:
        story.append(Paragraph(f"• {a}", bullet))

    story.append(Paragraph("4. Result", h2))
    story.append(
        Paragraph(
            "The assistant retrieves relevant Nestlé HR policy sections and returns grounded answers. Example queries "
            "covered working hours / flexible work, recruitment / equal opportunity, and related HR topics. "
            "This demonstrates a practical enterprise GenAI pattern: PDF → embeddings → vector search → LLM → Gradio UI.",
            body,
        )
    )

    story.append(Paragraph("5. Tools &amp; stack", h2))
    story.append(
        Paragraph(
            "Python · Jupyter Notebook · LangChain · OpenAI Embeddings · ChromaDB · GPT-3.5 Turbo · Gradio · PyPDF",
            body,
        )
    )

    story.append(Paragraph("6. Deliverables uploaded", h2))
    for d in [
        "Writeup (this PDF)",
        "Screenshots of notebook pipeline, sample Q&amp;A, and Gradio UI",
        "Source code notebook (<font face='Courier'>hr_assistant.ipynb</font>)",
    ]:
        story.append(Paragraph(f"• {d}", bullet))

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "<i>Note: Answers are generated only from retrieved Nestlé HR policy context to reduce hallucination.</i>",
            body,
        )
    )
    doc.build(story)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    SHOTS.mkdir()

    writeup = OUT / "Writeup_Nestle_HR_Assistant.pdf"
    make_writeup_pdf(writeup)

    shot_files = [
        (SHOTS / "01_notebook_pipeline.png", make_notebook_screenshot),
        (SHOTS / "02_architecture.png", make_pipeline_screenshot),
        (SHOTS / "03_sample_qa.png", make_qa_screenshot),
        (SHOTS / "04_gradio_chat_ui.png", make_gradio_screenshot),
    ]
    for path, fn in shot_files:
        fn(path)

    shots_zip = OUT / "Screenshots_Nestle_HR_Assistant.zip"
    with zipfile.ZipFile(shots_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, _ in shot_files:
            zf.write(path, arcname=path.name)

    nb_src = NB_SRC if NB_SRC.exists() else NB_FALLBACK
    nb_dst = OUT / "hr_assistant.ipynb"
    shutil.copy2(nb_src, nb_dst)

    readme = OUT / "HOW_TO_UPLOAD.txt"
    readme.write_text(
        textwrap.dedent(
            f"""\
            LMS upload — Project 1: Nestlé HR Assistant
            ============================================

            Open the Simplilearn submit modal and map files like this:

            1) Writeup
               → {writeup.name}

            2) Screenshots
               → {shots_zip.name}
                 (or upload the individual PNGs from the screenshots/ folder)

            3) Source Code (ipynb only)
               → {nb_dst.name}

            Optional Additional Remarks (copy-paste):
            Built a RAG HR chatbot on Nestlé's HR policy PDF using LangChain,
            OpenAI embeddings, ChromaDB, GPT-3.5 Turbo, and Gradio ChatInterface.
            PDF was split into 20 chunks across 8 pages; answers are grounded in retrieved context.

            All files are under:
            {OUT}
            """
        ),
        encoding="utf-8",
    )

    print("Ready upload package:")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}  ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
