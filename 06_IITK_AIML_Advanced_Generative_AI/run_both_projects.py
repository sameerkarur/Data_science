#!/usr/bin/env python3
"""Run both course-end projects and save outputs for submission."""

import json
import os
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

BASE = Path(__file__).resolve().parent
SECRETS = Path.home() / "Documents" / "opencareerai-secrets.json"
OUTPUT = BASE / "submission_outputs"
OUTPUT.mkdir(exist_ok=True)


def load_api_key() -> None:
    with open(SECRETS, encoding="utf-8") as f:
        os.environ["OPENAI_API_KEY"] = json.load(f)["ai_apis"]["openai_api_key"]
    print("✓ Loaded OpenAI API key from opencareerai-secrets.json")


def run_project1() -> None:
    print("\n" + "=" * 60)
    print("PROJECT 1: Nestlé HR Assistant (RAG)")
    print("=" * 60)

    os.chdir(BASE / "project1_hr_assistant")

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pdf_path = "Dataset/the_nestle_hr_policy_pdf_2012.pdf"
    chroma_dir = "chroma_hr_db"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} PDF pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} text chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()
    print("✓ ChromaDB vector store ready")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """You are Nestlé's HR policy assistant. Answer ONLY using the provided context.
If the answer is not in the context, say you don't have that information in the HR policy document.
Be concise, professional, and accurate.

Context:
{context}

Question: {input}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is Nestlé's policy on annual leave?",
        "What are the working hours guidelines?",
        "What is the recruitment policy?",
    ]

    results = []
    for q in questions:
        print(f"\nQ: {q}")
        a = qa_chain.invoke(q)
        print(f"A: {a[:500]}{'...' if len(a) > 500 else ''}")
        results.append({"question": q, "answer": a})

    out_file = OUTPUT / "project1_qa_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n✓ Saved QA results to {out_file}")


def run_project2() -> None:
    print("\n" + "=" * 60)
    print("PROJECT 2: Netflix Design Generator (DALL·E)")
    print("=" * 60)

    from openai import OpenAI

    client = OpenAI()

    def generate_image(prompt: str, size: str = "1024x1024") -> Image.Image:
        enhanced = (
            f"Professional Netflix-style streaming promotional poster design: {prompt.strip()}. "
            "Cinematic lighting, bold typography space, high contrast, modern digital marketing banner."
        )
        response = client.images.generate(
            model="gpt-image-1",
            prompt=enhanced,
            n=1,
            size=size,
        )
        item = response.data[0]
        if item.url:
            image_bytes = requests.get(item.url, timeout=60).content
        else:
            import base64
            image_bytes = base64.b64decode(item.b64_json)
        return Image.open(BytesIO(image_bytes))

    prompts = [
        "Dark thriller series poster with neon city skyline and mysterious silhouette",
        "Romantic drama series poster with couple under rain and warm city lights",
    ]

    for i, p in enumerate(prompts, 1):
        print(f"\nGenerating image {i}: {p[:60]}...")
        img = generate_image(p)
        out_path = OUTPUT / f"project2_design_{i}.png"
        img.save(out_path)
        print(f"✓ Saved {out_path} ({img.size[0]}x{img.size[1]})")


def main() -> int:
    load_api_key()
    try:
        run_project1()
    except Exception as e:
        print(f"✗ Project 1 failed: {e}", file=sys.stderr)
        return 1

    try:
        run_project2()
    except Exception as e:
        print(f"✗ Project 2 failed: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("ALL PROJECTS COMPLETED SUCCESSFULLY")
    print(f"Outputs: {OUTPUT}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
