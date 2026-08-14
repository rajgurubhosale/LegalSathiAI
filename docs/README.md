# LegalSaathi — Scope & Disclaimer

LegalSaathi provides general legal information about Indian criminal law under the
**Bharatiya Nyaya Sanhita (BNS)** and **Bharatiya Nagarik Suraksha Sanhita (BNSS)** only.

## This tool:
- Explains offences, punishments, and procedures as written in the BNS/BNSS.
- Cites the exact Section number for every claim it makes.

## This tool does NOT:
- Provide legal advice for your specific situation or case.
- Cover other laws (Constitution, civil law, IT Act, family law, etc.).
- Know your case facts, location, or any currently active legal orders.
- Predict how a court will rule in any real case.

## Example questions you can ask
- "What is the punishment for theft under BNS?"
- "Is murder a bailable offence?"
- "What does 'cognizable' mean?"
- "Which court handles cases of extortion?"

## Example questions outside scope
- Questions about laws other than BNS/BNSS.
- Advice specific to your personal situation or an ongoing case.
- Predictions about how a court will rule.

## Not legal advice
This is general information only, not a substitute for a licensed advocate.
For any real legal situation, please consult a qualified lawyer.

---

## Current evaluation metrics

Evaluated with DeepEval (contextual precision/recall/relevancy, answer relevancy,
faithfulness) against a golden-answer test set covering in-scope, general-definition,
and out-of-scope questions.

| Metric | Score |
|---|---|
| Faithfulness | 0.970 |
| Answer Relevancy | 0.779 |
| Recall | 0.662 |
| Precision | 0.675 |
| Relevancy | 0.472 |


---

## Roadmap / future work
Working on improvement.
- Improve reranking so the general sections
- [ ] Explore hybrid (dense + sparse/BM25) retrieval for exact section-number and
      keyword matches.
- [ ] Expand golden-answer eval set beyond the current sample for more reliable,
      lower-variance metrics.
- [ ] Expand coverage beyond BNS/BNSS to more Indian acts (e.g. IT Act, Motor Vehicles
      Act, POCSO, and other commonly referenced laws).

---

## Setup

### 1. Clone and create environment
```bash
git clone <your-repo-url>
cd LegalSaathi-AI
python -m venv legalAI
legalAI\Scripts\activate        # Windows
# source legalAI/bin/activate   # macOS/Linux
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, generate one from your working environment:
```bash
pip freeze > requirements.txt
```

Core libraries this project depends on:
```
pandas
numpy
langchain
langchain-openai
langchain-huggingface
langchain-chroma
langchain-text-splitters
chromadb
sentence-transformers
tiktoken
python-dotenv
deepeval
```

### 3. Set environment variables
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
MESH_DEEP_SEEK_FLASH=your_api_key_here
```

### 4. Build the data pipeline (first-time setup only)
Run in this order — each step depends on the previous one's output:
```bash
python "D:\LegalSaathi AI\src\chunking\pipeline.py"      # chunk BNS/BNSS source docs
python "D:\LegalSaathi AI\src\retrieval\parent_store.py" # merge parent data
# rebuild your Chroma vectorstore from the new chunks (embedding script)
```

---

## Running the CLI

Once setup is complete, run the interactive chat interface CLI:

```bash
python "D:\LegalSaathi AI\src\temp.py"
```

You'll see:
```
============================================================
LegalSaathi — Ask me about Indian criminal law (BNS/BNSS)
Type 'exit' or 'quit' to end the conversation.
============================================================

You: What is the punishment for theft under BNS?
```

Type your question and press Enter. Type `exit` or `quit` to end the session.

---

## Running evaluations

```bash
python "D:\LegalSaathi AI\src\evalutions\evals.py"
```

This scores the pipeline against the golden-answer CSV and saves per-row, per-metric
results to `eval_results.csv`.