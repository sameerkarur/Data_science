# Project 1 Writeup: AI-Powered HR Assistant (Nestlé)

## Situation
Nestlé's HR team needs faster access to policy information spread across long PDF documents. Employees often spend time searching manuals instead of getting instant answers.

## Task
Build a conversational chatbot that answers HR policy questions using Nestlé's HR PDF, OpenAI GPT-3.5 Turbo, LangChain RAG, ChromaDB embeddings, and a Gradio UI.

## Action
1. **Environment setup:** Configured OpenAI API credentials and installed LangChain, ChromaDB, PyPDF, and Gradio.
2. **Document ingestion:** Loaded `the_nestle_hr_policy_pdf_2012.pdf` with `PyPDFLoader` and split text using `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap).
3. **Vector store:** Generated OpenAI embeddings and stored chunks in ChromaDB for semantic retrieval.
4. **QA system:** Built a RAG pipeline with GPT-3.5 Turbo and a prompt template that restricts answers to retrieved context.
5. **UI deployment:** Launched a Gradio `ChatInterface` for interactive HR Q&A.

## Result
The assistant retrieves relevant HR policy sections and generates grounded answers. Example queries include leave policy, recruitment, working hours, and benefits. The solution improves HR self-service, reduces manual document lookup, and demonstrates practical enterprise GenAI deployment.

## Key Architecture Visualizations
- Notebook cells showing PDF load + chunk count
- Vector store creation output
- Sample QA response in notebook
- Gradio chatbot UI with 2–3 example questions and answers
