# Codemate-Assignment2
This project is a local research assistant that ingests your documents, breaks them into chunks, and creates embeddings to store their meaning. It retrieves relevant parts when you ask a question, performs multi-step reasoning, and summarizes the results into clear reports you can export as Markdown or PDF.
# Local Research Agent

## 📖 Overview
The Local Research Agent is an **offline research assistant** that helps you explore and understand your own documents. It ingests PDFs, Markdown, or text files, breaks them into smaller chunks, generates embeddings (semantic fingerprints), and stores them for fast retrieval. When you ask a question, it finds relevant passages, performs multi-step reasoning, and summarizes the results into a clean research report. Reports can be exported as **Markdown** or **PDF**.

---

## ✨ Features
- Ingests local documents (`.pdf`, `.txt`, `.md`)
- Splits text into manageable chunks
- Generates embeddings using **sentence-transformers**
- Stores embeddings in **FAISS** and metadata in **SQLite**
- Handles multi-step queries by decomposing them into sub-questions
- Summarizes results into structured reports
- Exports research reports in **Markdown** and **PDF**
- Runs completely **offline**, no external APIs required

---

## ⚙️ Installation
Install the required Python packages:
```bash
pip install sentence-transformers faiss-cpu sqlalchemy nltk PyPDF2 transformers torch reportlab tqdm
```

---

## 🚀 Usage

### 1. Ingest documents
Place your files in a folder (e.g., `docs/`) and ingest them:
```bash
python local_research_agent.py --ingest docs
```

### 2. Run a query
Ask a question based on the ingested documents:
```bash
python local_research_agent.py --query "What are the challenges in renewable energy storage?" --export-md report.md --export-pdf report.pdf
```

### 3. View the results
- `report.md` → Markdown research report
- `report.pdf` → PDF research report

---

## 📂 Example Documents
You can test with your own files or create simple examples like:
- **file1.txt**: Notes on climate change causes
- **file2.txt**: Summary of renewable energy sources
- **file3.txt**: Basics of artificial intelligence

Then ask:
```bash
python local_research_agent.py --query "What are the effects of climate change and how does renewable energy help?" --export-md report.md --export-pdf report.pdf
```

---

## 🧠 How It Works
1. **Read documents** → Extracts text from PDFs or TXT/MD files
2. **Chunk text** → Splits into smaller, meaningful sections
3. **Embed** → Creates numerical fingerprints with SentenceTransformer
4. **Store** → Saves embeddings in FAISS + metadata in SQLite
5. **Query** → Breaks your question into sub-queries and searches
6. **Summarize** → Produces a structured research report
7. **Export** → Saves the report as Markdown or PDF

---

## 🎯 Why Use It?
- Works completely offline, ensuring data privacy
- Turns your personal library into a searchable knowledge base
- Saves time by summarizing and organizing key findings
- Flexible for academic research, technical docs, or study notes

---

## 📌 License
This project is provided for educational and personal research purposes. You are free to modify and extend it for your needs.
