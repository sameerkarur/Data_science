import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#64748b"))
        page_text = f"Sameer Karur  |  Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 0.5 * inch, 0.35 * inch, page_text)
        self.restoreState()

def build_pdf(filename):
    pdf = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch
    )

    PRIMARY = colors.HexColor("#0f172a")      # Slate 900
    ACCENT = colors.HexColor("#0369a1")       # Sky 700 / Professional blue
    TEXT_DARK = colors.HexColor("#1e293b")    # Slate 800
    TEXT_MUTED = colors.HexColor("#475569")   # Slate 600
    LINE_COLOR = colors.HexColor("#cbd5e1")   # Slate 300

    styles = getSampleStyleSheet()

    name_style = ParagraphStyle(
        'DocName',
        fontName='Helvetica-Bold',
        fontSize=21,
        leading=23,
        textColor=PRIMARY,
        alignment=0,
        spaceAfter=2
    )
    
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13.5,
        textColor=ACCENT,
        alignment=0,
        spaceAfter=4
    )

    contact_style = ParagraphStyle(
        'DocContact',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=TEXT_MUTED,
        alignment=0,
        spaceAfter=6
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=PRIMARY,
        spaceBefore=7,
        spaceAfter=2
    )

    item_title = ParagraphStyle(
        'ItemTitle',
        fontName='Helvetica-Bold',
        fontSize=9.2,
        leading=12,
        textColor=TEXT_DARK
    )

    item_subtitle = ParagraphStyle(
        'ItemSubtitle',
        fontName='Helvetica-Oblique',
        fontSize=8.2,
        leading=11,
        textColor=TEXT_MUTED,
        alignment=2
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        fontName='Helvetica',
        fontSize=8.3,
        leading=11.5,
        textColor=TEXT_DARK,
        leftIndent=11,
        firstLineIndent=-7,
        spaceAfter=2.5
    )

    summary_style = ParagraphStyle(
        'SummaryText',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=TEXT_DARK,
        spaceAfter=3
    )

    skills_label = ParagraphStyle(
        'SkillsLabel',
        fontName='Helvetica-Bold',
        fontSize=8.3,
        leading=11.5,
        textColor=PRIMARY
    )

    skills_val = ParagraphStyle(
        'SkillsVal',
        fontName='Helvetica',
        fontSize=8.3,
        leading=11.5,
        textColor=TEXT_DARK
    )

    story = []

    # 1. Header
    story.append(Paragraph("SAMEER KARUR", name_style))
    story.append(Paragraph("Data Analyst & Technical Consultant | Applied AI & Machine Learning Practitioner", title_style))
    
    contact_text = (
        "<b>Email:</b> sameerkarur333@gmail.com &nbsp;|&nbsp; "
        "<b>Phone:</b> +91 6360053030 &nbsp;|&nbsp; "
        "<b>Location:</b> Bengaluru, Karnataka, India<br/>"
        "<b>LinkedIn:</b> <a href='https://www.linkedin.com/in/sameer-karur-a5648224b/' color='#0369a1'>linkedin.com/in/sameer-karur-a5648224b</a> &nbsp;|&nbsp; "
        "<b>GitHub Portfolio:</b> <a href='https://github.com/sameerkarur/Data_science' color='#0369a1'>github.com/sameerkarur/Data_science</a>"
    )
    story.append(Paragraph(contact_text, contact_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=0, spaceAfter=4))

    # 2. Professional Summary
    story.append(Paragraph("PROFESSIONAL SUMMARY", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LINE_COLOR, spaceBefore=0, spaceAfter=4))
    summary = (
        "<b>Data Analyst & Technical Consultant</b> with 2.5+ years of experience transforming complex enterprise datasets into actionable business intelligence, automated analytical pipelines, and mission-critical enterprise systems. Proven track record spanning high-scale consumer e-commerce fulfillment (<b>Flipkart</b>) and global life sciences & pharmacovigilance technology (<b>ArisGlobal</b>). Advanced hands-on expertise in SQL, Python, and Tableau, complemented by specialized rigor in Generative AI (RAG, LangChain, Vector Databases), Deep Learning (TensorFlow, Computer Vision), and Predictive Machine Learning through <b>E&ICT Academy, IIT Kanpur</b>."
    )
    story.append(Paragraph(summary, summary_style))

    # 3. Technical Skills
    story.append(Paragraph("TECHNICAL SKILLS & COMPETENCIES", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LINE_COLOR, spaceBefore=0, spaceAfter=4))

    skills_data = [
        [Paragraph("<b>Analytics & BI:</b>", skills_label), Paragraph("Advanced SQL (Window Functions, CTEs, Complex Joins), Tableau Dashboard Design, Exploratory Data Analysis (EDA), KPI Tracking, Root Cause Analysis, Statistical Hypothesis Testing, Supply Chain & Operations Analytics", skills_val)],
        [Paragraph("<b>Programming & DB:</b>", skills_label), Paragraph("Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn), JavaScript, MySQL, Relational Database Management (RDBMS), Data Extraction & Automated ETL Pipelines", skills_val)],
        [Paragraph("<b>Machine Learning:</b>", skills_label), Paragraph("Supervised & Unsupervised Learning, XGBoost, Random Forest, Logistic Regression, K-Means Clustering, SMOTE Imbalance Handling, Time-Series Demand Forecasting", skills_val)],
        [Paragraph("<b>Deep Learning & CV:</b>", skills_label), Paragraph("TensorFlow, Keras, Convolutional Neural Networks (CNNs), Transfer Learning (MobileNetV2), Computer Vision (OpenCV), Multi-Class Classification & Object Localization", skills_val)],
        [Paragraph("<b>Generative AI:</b>", skills_label), Paragraph("Retrieval-Augmented Generation (RAG), LangChain, ChromaDB Vector Database, OpenAI APIs, Prompt Engineering, Gradio Web UI Deployment", skills_val)],
        [Paragraph("<b>Cloud & Developer Tools:</b>", skills_label), Paragraph("Microsoft Azure (AZ-900 Certified), Git, GitHub, Jupyter Notebook, VS Code, Advanced Excel", skills_val)],
    ]
    t = Table(skills_data, colWidths=[1.4 * inch, 6.1 * inch])
    t.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t)

    # 4. Professional Experience
    story.append(Spacer(1, 4))
    story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LINE_COLOR, spaceBefore=0, spaceAfter=5))

    # Role 1: ArisGlobal
    exp1_header = [
        [Paragraph("<b>Consultant</b> — ArisGlobal", item_title), Paragraph("Bengaluru, India &nbsp;|&nbsp; Apr 2026 – Present", item_subtitle)]
    ]
    t_exp1 = Table(exp1_header, colWidths=[4.7 * inch, 2.8 * inch])
    t_exp1.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_exp1)
    
    bullets_aris = [
        "<b>Regulatory Data Engineering:</b> Configure business validation rules, automated regulatory workflow logic, and data verification scripts using SQL and JavaScript for enterprise Pharmacovigilance (PV) safety platforms.",
        "<b>Data Integrity & Compliance:</b> Query, audit, and validate multi-tiered clinical and adverse event (AE) databases to guarantee data completeness and strict adherence to global health regulatory standards (FDA, EMA).",
        "<b>Automation & ETL Pipelines:</b> Develop Python and SQL automation scripts for structured data extraction, schema validation, and reporting consistency across high-throughput client safety systems.",
        "<b>Stakeholder Consulting:</b> Interface directly with cross-functional technical teams and client domain leads to troubleshoot mission-critical data anomalies, optimize data flows, and accelerate case processing throughput."
    ]
    for b in bullets_aris:
        story.append(Paragraph(f"• {b}", bullet_style))

    story.append(Spacer(1, 2))

    # Role 2: Flipkart
    exp2_header = [
        [Paragraph("<b>Data Analyst</b> — Flipkart (Payroll: RLabs Enterprise Services)", item_title), Paragraph("Bengaluru, India &nbsp;|&nbsp; Jan 2024 – Apr 2026", item_subtitle)]
    ]
    t_exp2 = Table(exp2_header, colWidths=[4.7 * inch, 2.8 * inch])
    t_exp2.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_exp2)

    bullets_flipkart = [
        "<b>Executive BI & Dashboards:</b> Architected and maintained executive Tableau dashboards monitoring 10+ operational KPIs (SLA compliance, fulfillment throughput, breach rates, delivery TAT) consumed daily by business and city operations leadership.",
        "<b>Big Data SQL Analytics:</b> Authored high-performance SQL queries (window functions, recursive CTEs, complex joins) across multi-million row transactional databases to conduct root-cause investigations on operational bottlenecks.",
        "<b>Process Automation:</b> Built automated data extraction, cleaning, and reporting pipelines in Python (Pandas, NumPy), slashing recurring manual reporting turnaround time by over 70% and saving 10+ engineering hours weekly.",
        "<b>Cross-Functional Operations Impact:</b> Partnered with warehouse, supply chain, and logistics leadership to translate ambiguous fulfillment delays into structured metrics, directly informing shift scheduling, capacity planning, and resource allocation.",
        "<b>Data Quality Governance:</b> Built automated discrepancy audits reconciling warehouse log systems with downstream reporting tables, ensuring 99%+ reporting reliability."
    ]
    for b in bullets_flipkart:
        story.append(Paragraph(f"• {b}", bullet_style))

    # 5. Key Projects Portfolio (Starts cleanly on Page 2)
    from reportlab.platypus import PageBreak
    story.append(PageBreak())
    story.append(Paragraph("FEATURED AI & MACHINE LEARNING PROJECTS", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LINE_COLOR, spaceBefore=0, spaceAfter=5))

    projects = [
        (
            "AI-Powered Enterprise HR Assistant (RAG System & Gradio UI)",
            "Python, LangChain, ChromaDB, OpenAI Embeddings, GPT-3.5 Turbo, Gradio",
            [
                "Architected a production-ready conversational RAG assistant grounded in official multi-page PDF policies to resolve employee HR inquiries in natural language.",
                "Chunked documents with RecursiveCharacterTextSplitter and generated vector embeddings persisted in ChromaDB with cosine similarity retrieval.",
                "Enforced strict contextual grounding prompts that eliminated hallucinations, deploying an interactive Gradio web chat interface."
            ]
        ),
        (
            "Autonomous Driving: Vehicle Perception & Tesla Crash Analytics",
            "TensorFlow, Keras, MobileNetV2, OpenCV, Scikit-learn, Pandas, Seaborn",
            [
                "Trained a transfer-learning MobileNetV2 crop classifier on 5,600+ road scene images with 17,900+ bounding boxes, achieving <b>89.9% coarse classification accuracy</b> across multiple vehicle classes.",
                "Conducted deep safety analytics on 293 fatal Tesla crash records (352 deaths), analyzing victim roles, collision mechanics, model distributions, and empirical verification of Autopilot-involved incidents."
            ]
        ),
        (
            "Multi-Restaurant Demand & Sales Forecasting System",
            "Python, XGBoost, Random Forest, Scikit-learn, Pandas, Time-Series Modeling",
            [
                "Analyzed 109,600 item-day transaction records across 6 restaurants and 100 menu items to forecast daily demand.",
                "Engineered time-series calendar features (seasonality, day-of-week, quarter, revenue vs. volume velocity).",
                "Built and evaluated Linear Regression, Random Forest, and XGBoost; champion XGBoost achieved <b>RMSE 57.90 (R² 0.948)</b> on a 6-month test holdout, generating an accurate full-year forward demand forecast."
            ]
        ),
        (
            "Preserving Cultural Heritage: Monument Classification & Recommender",
            "TensorFlow, Keras, MobileNetV2, Scikit-surprise (SVD), OpenCV, Pandas",
            [
                "Developed a dual-engine AI platform combining Deep Learning computer vision with an intelligent tourism recommendation system.",
                "Built a MobileNetV2 classifier across 10 global heritage landmarks, achieving <b>86.5% test accuracy</b> with data augmentation.",
                "Engineered a personalized collaborative filtering recommender with SVD and cosine similarity, serving top-N customized itineraries."
            ]
        ),
        (
            "Generative AI Marketing Campaign Asset & Pitch Generator",
            "Python, OpenAI DALL-E / Images API, GPT-4 / ChatGPT, Gradio UI, PIL",
            [
                "Built an end-to-end GenAI application that takes natural-language brand concepts and generates professional marketing pitch copy and 4K campaign imagery.",
                "Designed chained system prompts combining visual prompt expansion with structured marketing copy generation.",
                "Deployed an intuitive Gradio web UI allowing non-technical marketing teams to prototype multi-channel ad concepts in seconds."
            ]
        ),
        (
            "Predictive Employee Turnover Analytics & Retention Modeling",
            "Python, Scikit-learn, Random Forest, SMOTE, K-Means Clustering, Seaborn",
            [
                "Modeled attrition patterns across 15,000 employee profiles; clustered key dissatisfaction personas via K-Means.",
                "Resolved severe class imbalance via SMOTE resampling; achieved <b>97.2% accuracy and 0.98 ROC-AUC</b> with Random Forest, segmenting staff into prioritized retention risk tiers for targeted HR action."
            ]
        )
    ]

    for p_title, p_tech, p_bullets in projects:
        p_table = Table([
            [Paragraph(f"<b>{p_title}</b>", item_title), Paragraph(f"<i>{p_tech}</i>", item_subtitle)]
        ], colWidths=[4.7 * inch, 2.8 * inch])
        p_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
            ('TOPPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(p_table)
        for b in p_bullets:
            story.append(Paragraph(f"• {b}", bullet_style))
        story.append(Spacer(1, 1))

    # 6. Education & Certifications
    story.append(Spacer(1, 4))
    story.append(Paragraph("EDUCATION & CERTIFICATIONS", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LINE_COLOR, spaceBefore=0, spaceAfter=5))

    edu_table_data = [
        [
            Paragraph("<b>Professional Certificate in Generative AI and Machine Learning</b><br/>E&ICT Academy, IIT Kanpur (Dec 2025 – Nov 2026)", item_title),
            Paragraph("Top Capstone Delivery & Distinction<br/>10+ Real-World AI Implementations", item_subtitle)
        ],
        [
            Paragraph("<b>Bachelor of Engineering (B.E.) in Electronics & Communication</b><br/>Vishwanathrao Deshpande Institute of Technology, Karnataka (Graduated: Aug 2022)", item_title),
            Paragraph("First Class with Distinction<br/>Technical Foundations", item_subtitle)
        ]
    ]
    t_edu = Table(edu_table_data, colWidths=[5.0 * inch, 2.5 * inch])
    t_edu.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_edu)

    story.append(Spacer(1, 2))
    certs = (
        "<b>Professional Certifications:</b> Microsoft Certified: Azure Fundamentals (AZ-900) &nbsp;|&nbsp; "
        "SQL for Business Analytics &nbsp;|&nbsp; Python for Data Analysis &nbsp;|&nbsp; Exploratory Data Analysis with Tableau"
    )
    story.append(Paragraph(certs, summary_style))

    pdf.build(story, canvasmaker=NumberedCanvas)
    print("PDF build successful.")

if __name__ == "__main__":
    out_pdf = Path("Sameer_Karur_Resume.pdf")
    build_pdf(str(out_pdf))
