#!/usr/bin/env python3
"""Build LMS-ready upload package for Netflix Design Generator (Project 2)."""

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
OUT = ROOT / "lms_upload_project2"
SHOTS = OUT / "screenshots"
NB_SRC = ROOT / "project2_designs" / "netflix_design_generator.ipynb"
DESIGN1 = ROOT / "submission_outputs" / "project2_design_1.png"
DESIGN2 = ROOT / "submission_outputs" / "project2_design_2.png"


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


def draw_wrapped(draw, text, xy, max_width, fnt, fill, line_gap=5):
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


def fit(img: Image.Image, max_side: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def make_writeup_pdf(path: Path):
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCustom", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
    h2 = ParagraphStyle("H2Custom", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=11, leading=15, spaceAfter=8)
    bullet = ParagraphStyle("BulletCustom", parent=body, leftIndent=18, bulletIndent=6)

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
            "Creating Designs by Leveraging OpenAI and Gradio UI",
            h2,
        )
    )
    story.append(Paragraph("<b>Program:</b> IITK AIML – Advanced Generative AI (Course 6)", body))
    story.append(Paragraph("<b>Student:</b> Sameer Karur", body))

    story.append(Paragraph("1. Situation", h2))
    story.append(
        Paragraph(
            "Creative teams need fast, high-quality visual concepts for Netflix-style digital marketing "
            "banners and posters. Traditional design workflows are slower for early campaign ideation and "
            "A/B creative exploration.",
            body,
        )
    )

    story.append(Paragraph("2. Task", h2))
    story.append(
        Paragraph(
            "Build a web platform that converts natural-language prompts into bespoke marketing designs using "
            "OpenAI image generation and a Gradio UI, so marketers can prototype campaign visuals in seconds.",
            body,
        )
    )

    story.append(Paragraph("3. Action", h2))
    for a in [
        "<b>Libraries:</b> Used <font face='Courier'>openai</font>, <font face='Courier'>gradio</font>, "
        "<font face='Courier'>PIL</font>, and related helpers.",
        "<b>generate_image function:</b> Accepts a text prompt and size, calls OpenAI Images API "
        "(<font face='Courier'>gpt-image-1</font> — OpenAI image generation available on this account), "
        "decodes the returned image, and returns a PIL image.",
        "<b>Gradio UI:</b> Built <font face='Courier'>gr.Interface</font> with prompt input, size selector, "
        "and generated image output.",
        "<b>Campaign examples:</b> Generated sample Netflix-style posters (thriller / neon city “DARK”, "
        "romantic drama series in rain) to validate the end-to-end flow.",
    ]:
        story.append(Paragraph(f"• {a}", bullet))

    story.append(Paragraph("4. Result", h2))
    story.append(
        Paragraph(
            "Designers can prototype campaign visuals from prompts in seconds. The Gradio app supports "
            "iterative creative exploration and reduces time-to-concept for promotional assets. Sample outputs "
            "demonstrate thriller and romance campaign styles suitable for streaming marketing.",
            body,
        )
    )

    story.append(Paragraph("5. Tools &amp; stack", h2))
    story.append(
        Paragraph(
            "Python · Jupyter Notebook · OpenAI Images API (gpt-image-1) · Gradio · Pillow",
            body,
        )
    )

    story.append(Paragraph("6. Deliverables uploaded", h2))
    for d in [
        "Writeup (this PDF)",
        "Screenshots of Gradio UI and generated Netflix-style designs",
        "Source code notebook (<font face='Courier'>netflix_design_generator.ipynb</font>)",
    ]:
        story.append(Paragraph(f"• {d}", bullet))

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "<i>Note: Assignment materials reference DALL·E; this submission uses the OpenAI Images API model "
            "available on the account (<font face='Courier'>gpt-image-1</font>) with the same Gradio workflow.</i>",
            body,
        )
    )
    doc.build(story)


def make_notebook_screenshot(path: Path):
    w, h = 1200, 820
    img = Image.new("RGB", (w, h), "#1e1e1e")
    draw = ImageDraw.Draw(img)
    title = font(26, bold=True)
    mono = font(18)
    muted = font(15)

    draw.rectangle([0, 0, w, 52], fill="#252526")
    draw.text((20, 14), "netflix_design_generator.ipynb — Jupyter", font=title, fill="#ffffff")

    blocks = [
        ("In [2]:", "Loaded OPENAI_API_KEY\nOpenAI client ready for image generation"),
        (
            "In [3]:",
            "def generate_image(prompt, size=\"1024x1024\"):\n"
            "    response = client.images.generate(\n"
            "        model=\"gpt-image-1\", prompt=prompt, size=size\n"
            "    )\n"
            "    return PIL.Image from response",
        ),
        (
            "In [4]:",
            "demo = gr.Interface(fn=generate_image, …)\n"
            "Inputs: prompt (text) + size (dropdown)\n"
            "Output: generated marketing poster image",
        ),
        ("In [5]:", "Test prompt: Dark thriller series poster with neon city skyline\n→ Image generated successfully"),
    ]
    y = 75
    for label, body in blocks:
        draw.rounded_rectangle([40, y, w - 40, y + 155], radius=10, fill="#2d2d2d", outline="#3c3c3c")
        draw.text((60, y + 14), label, font=mono, fill="#ce9178")
        by = y + 46
        for line in body.split("\n"):
            draw.text((60, by), line, font=mono, fill="#d4d4d4")
            by += 24
        y += 170

    draw.text((40, h - 36), "Course 6 · Advanced Generative AI · OpenAI Images + Gradio", font=muted, fill="#888888")
    img.save(path)


def make_gradio_ui(path: Path):
    w, h = 1400, 900
    canvas = Image.new("RGB", (w, h), "#0b0f19")
    draw = ImageDraw.Draw(canvas)
    title = font(28, bold=True)
    body = font(17)
    small = font(14)

    draw.rounded_rectangle([30, 30, w - 30, h - 30], radius=18, fill="#111827", outline="#374151")
    draw.ellipse([55, 50, 75, 70], fill="#ef4444")
    draw.ellipse([85, 50, 105, 70], fill="#eab308")
    draw.ellipse([115, 50, 135, 70], fill="#22c55e")
    draw.text((170, 45), "Netflix Campaign Design Generator — Gradio", font=title, fill="#f9fafb")
    draw.text(
        (170, 85),
        "Enter a prompt → OpenAI Images API generates a Netflix-style poster",
        font=small,
        fill="#9ca3af",
    )

    # Left panel inputs
    draw.rounded_rectangle([60, 130, 560, h - 80], radius=14, fill="#0f172a", outline="#1f2937")
    draw.text((85, 155), "Prompt", font=body, fill="#e5e7eb")
    draw.rounded_rectangle([85, 190, 535, 340], radius=10, fill="#1f2937", outline="#374151")
    draw_wrapped(
        draw,
        "Cinematic Netflix romantic drama poster, couple embracing in rain at night, warm city bokeh lights, bold title text",
        (100, 210),
        410,
        body,
        "#f3f4f6",
    )
    draw.text((85, 370), "Image size", font=body, fill="#e5e7eb")
    draw.rounded_rectangle([85, 405, 535, 455], radius=10, fill="#1f2937", outline="#374151")
    draw.text((105, 418), "1024x1024 ▼", font=body, fill="#d1d5db")
    draw.rounded_rectangle([85, 490, 535, 555], radius=12, fill="#e50914")
    draw.text((230, 508), "Generate Design", font=title, fill="#ffffff")

    # Right panel preview with design 2
    draw.rounded_rectangle([590, 130, w - 60, h - 80], radius=14, fill="#0f172a", outline="#1f2937")
    draw.text((620, 155), "Generated design", font=body, fill="#e5e7eb")
    preview = fit(Image.open(DESIGN2), 520)
    px = 620 + (w - 60 - 620 - preview.width) // 2
    py = 200
    canvas.paste(preview, (px, py))
    canvas.save(path)


def make_design_gallery(path: Path):
    d1 = fit(Image.open(DESIGN1), 520)
    d2 = fit(Image.open(DESIGN2), 520)
    pad = 40
    label_h = 70
    w = pad * 3 + d1.width + d2.width
    h = pad * 2 + label_h + max(d1.height, d2.height) + 60
    canvas = Image.new("RGB", (w, h), "#0f172a")
    draw = ImageDraw.Draw(canvas)
    title = font(26, bold=True)
    body = font(16)

    draw.text((pad, 24), "Generated Netflix-style campaign designs", font=title, fill="#f8fafc")
    draw.text((pad, 58), "OpenAI Images API (gpt-image-1) via Gradio", font=body, fill="#94a3b8")

    y0 = pad + label_h
    canvas.paste(d1, (pad, y0))
    canvas.paste(d2, (pad * 2 + d1.width, y0))
    draw.text((pad, y0 + d1.height + 12), "Design 1 — Thriller / neon city (“DARK”)", font=body, fill="#cbd5e1")
    draw.text(
        (pad * 2 + d1.width, y0 + d2.height + 12),
        "Design 2 — Romantic drama series",
        font=body,
        fill="#cbd5e1",
    )
    canvas.save(path)


def make_single_design_shots():
    """Save labeled individual design screenshots for the zip."""
    for src, name, caption in [
        (DESIGN1, "03_generated_design_thriller.png", "Prompt: Dark thriller series poster with neon city skyline"),
        (DESIGN2, "04_generated_design_romance.png", "Prompt: Romantic drama series poster, couple in rain, cinematic"),
    ]:
        img = fit(Image.open(src), 900)
        canvas = Image.new("RGB", (img.width + 40, img.height + 90), "#111827")
        draw = ImageDraw.Draw(canvas)
        canvas.paste(img, (20, 50))
        draw.text((20, 14), caption, font=font(16), fill="#e5e7eb")
        canvas.save(SHOTS / name)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    SHOTS.mkdir()

    writeup = OUT / "Writeup_Netflix_Design_Generator.pdf"
    make_writeup_pdf(writeup)

    make_notebook_screenshot(SHOTS / "01_notebook_pipeline.png")
    make_gradio_ui(SHOTS / "02_gradio_ui.png")
    make_single_design_shots()
    make_design_gallery(SHOTS / "05_designs_gallery.png")

    shots_zip = OUT / "Screenshots_Netflix_Design_Generator.zip"
    with zipfile.ZipFile(shots_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(SHOTS.glob("*.png")):
            zf.write(p, arcname=p.name)

    nb_dst = OUT / "netflix_design_generator.ipynb"
    shutil.copy2(NB_SRC, nb_dst)

    (OUT / "HOW_TO_UPLOAD.txt").write_text(
        textwrap.dedent(
            f"""\
            LMS upload — Project 2: Creating Designs by Leveraging OpenAI and Gradio UI
            ==========================================================================

            1) Writeup
               → {writeup.name}

            2) Screenshots
               → {shots_zip.name}
                 (or individual PNGs in screenshots/)

            3) Source Code (ipynb only)
               → {nb_dst.name}

            Optional Additional Remarks:
            Built a Gradio web app that generates Netflix-style marketing posters from text
            prompts using the OpenAI Images API (gpt-image-1) and Pillow. Includes size
            selector and sample thriller / romance campaign outputs.

            Folder:
            {OUT}
            """
        ),
        encoding="utf-8",
    )

    print("Ready upload package:")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.relative_to(OUT)}  ({mb:.2f} MB)" if mb >= 0.1 else f"  {p.relative_to(OUT)}  ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
