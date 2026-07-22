# Multi-Agent Paddy Disease Diagnostic and Fertilizer Recommendation System for Sri Lankan Farmers

**Module:** IT41043 - Agentic AI (Horizon Campus)  
**Branch:** `feature/rag-pipeline`

---

## 📌 Project Overview
An intelligent multi-agent system designed to assist Sri Lankan farmers and agricultural officers in diagnosing paddy plant diseases, recommending optimal fertilizer regimens (suited for Yala and Maha seasons), and retrieving authoritative agricultural knowledge from local policy guidelines, research papers, and technical bulletins.

---

## 🛠 Tech Stack & RAG Architecture (Step 2 Implementation)

### 1. Document Loading & Organization
The domain knowledge base consists of 20+ PDF documents organized into 6 logical subdirectories inside `Data/PDF/`:
- `Fertilizer_and_Chemicals`: Soil nutrient recommendations, NPK ratios, weedicides/pesticides.
- `General_Cultivation_Guidelines`: Seasonal planting guidelines (Yala/Maha), water management, land preparation.
- `Imports_Quarantine`: Import restrictions, quarantine acts, biosecurity measures.
- `Paddy_Diseases_and_Pests`: Detailed diagnostic manuals for Paddy Blast, Sheath Blight, Brown Planthopper, etc.
- `Paddy_Seed_Production`: Certification criteria, seed purity rates, germination standards.
- `Policy_and_Acts`: National agricultural acts, seed acts, environmental regulations.

### 2. Document Chunking Strategy
- **Text Splitter:** `RecursiveCharacterTextSplitter`
- **Chunk Size:** `1000` characters (~150–200 words)
- **Chunk Overlap:** `200` characters
- **Separators:** `["\n\n", "\n", ". ", " ", ""]`

#### 📖 Chunking Strategy Rationale (Academic Documentation)
- **Context Preservation:** Paddy disease symptoms (e.g., lesions, discoloration) and chemical dosage instructions often span multiple sentences or tabular text blocks. A chunk size of 1000 characters provides sufficient contextual window for dense domain-specific information without fragmenting chemical names or diagnosis steps.
- **Boundary Safety via Overlap:** The 200-character chunk overlap prevents critical facts positioned at page or paragraph boundaries from being severed across adjacent chunks, maintaining continuity for semantic retrieval.
- **Recursive Splitting:** Splitting hierarchically by double line breaks (`\n\n`), single line breaks (`\n`), and sentence periods (`. `) preserves paragraph integrity and prevents mid-sentence breaks wherever possible.

### 3. Multilingual Embedding Model
- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via HuggingFace Embeddings.
- **Rationale:** Sri Lankan agricultural publications and extension bulletins often contain a blend of English technical terms (e.g., *Pyricularia oryzae*, *NPK 14:14:14*) and Sinhala descriptions (e.g., *ගොයම් මැස්සා, පාළු රෝගය*). The 12-layer multilingual transformer projects Sinhala script and English query terms into a shared high-dimensional vector space, facilitating cross-lingual semantic search.

### 4. Local Vector Store
- **Vector Database:** FAISS (Facebook AI Similarity Search) / Chroma
- **Index Directory:** `./faiss_db/`
- **Metadata Retained:** `category`, `filename`, `source`, `page`.

---

## 🚀 Setup & Execution Guide

### Prerequisite Installation
```bash
pip install -r requirements.txt
```

### Running the RAG Pipeline & Retrieval Evaluation
Execute the RAG script to index all PDFs and evaluate retrieval on 5 domain-specific queries:
```bash
python rag_pipeline.py
```

### 🧪 Sample Evaluation Queries Included
1. *"What are the common symptoms and control measures for Paddy Blast disease?"*
2. *"What is the recommended NPK fertilizer application rate for Yala season paddy cultivation?"*
3. *"ගොයම් පාළු රෝගය පාලනය කරන්නේ කෙසේද?"* (Sinhala language query)
4. *"What are the quality standards and germination requirements for certified seed paddy in Sri Lanka?"*
5. *"What are the quarantine regulations for importing paddy seed and agricultural items into Sri Lanka?"*
