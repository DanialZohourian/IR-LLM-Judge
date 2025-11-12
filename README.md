# RAG Retrieval/Reranking Evaluator

A lightweight, LLM‑based evaluation framework to assess **retrieval** and **reranking** quality for Retrieval‑Augmented Generation (RAG). Given a set of queries and their retrieved (and optionally reranked) documents, the framework asks an LLM to judge whether **an answer is present** in the top‑k results and computes metrics such as **Answer Presence@k**, **Precision@k**, **Recall@k**, and **Accuracy@k**.

---

## ✨ Why this matters
RAG works only if your retriever hands the generator at least one **answerable** chunk. This repo measures exactly that: *“What’s the probability that at least one relevant document appears in the top‑k?”* — a practical proxy for end‑to‑end success.

---

## 🧪 Evaluation Method
For each query in your test set:
1. **Retrieve & rerank** with your chosen embedding model and reranker.
2. **Label relevance** of the top‑k docs with an LLM (binary: **YES/NO** = answer present or not).
3. **Compute metrics**, notably **Answer Presence@k** (1 if any relevant doc is in top‑k, else 0).

Average **Answer Presence@k** across queries to estimate how often your system provides at least one answerable document — crucial for robust RAG.

---

## 🧰 Evaluator Class
The `Evaluator` class calls an OpenAI‑compatible API (e.g., `gpt-4o-mini`, `gpt-4.1-mini`) to label each document as relevant or not.

```python
class Evaluator:
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o-mini", prompt_template: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.prompt_template = prompt_template or """Here are a question and a retrieved passage from a text corpus from the same domain as the question.

Can you judge whether an answer to the question can be derived from the retrieved passage, simply answer either “YES” or “NO”.

<binary>

Question: {query}; Retrieved Passage: {document}"""
        self.relevance_dict = {}  # stores latest document->label
        self.relevance_labels = []  # stores latest labels only

    def get_relevance_labels(self, docs: list[dict], query: str, top_k=None) -> dict:
        """
        Use GPT-4o Mini to determine if each document is relevant to the query.
        Stores and returns a dictionary of {'document_text': 1 or 0}
        """
        self.relevance_dict = {}
        for i, doc in enumerate(docs[:top_k]):
            prompt = self.prompt_template.format(query=query, document=doc["document"])

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1
            )

            answer = response.choices[0].message.content.strip().upper()
            label = 1 if "YES" in answer else 0
            self.relevance_dict[doc["document"]] = label

            print(f"✅ [{i+1}/{len(docs[:top_k])}] Relevance: {answer}")

        self.relevance_labels = list(self.relevance_dict.values())
        return self.relevance_dict

    def compute_metrics(self, ks: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], show = False) -> dict:
        """
        Compute metrics using the most recent relevance labels.
        """
        if not self.relevance_labels:
            raise ValueError("No relevance labels available. Run get_relevance_labels() first.")

        metrics = {}
        total_relevant = sum(self.relevance_labels)

        for k in ks:
            topk = self.relevance_labels[:k]
            tp = sum(topk)

            precision = tp / k if k else 0
            recall = tp / total_relevant if total_relevant else 0
            accuracy = tp / k if k else 0
            binary_acc = 0 if tp == 0 else 1

            metrics[f"@{k}"] = {
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "Accuracy": round(accuracy, 3),
                "Number of Relevant Docs": tp,
                "Answer Presence": binary_acc
            }

        if show:
            for k, vals in metrics.items():
                print(f"\nMetrics {k}:")
                for metric, val in vals.items():
                    print(f"{metric}: {val}")

        return metrics
```

> **Note**
> - `Accuracy` here is equivalent to `Precision` (TP/k) given the binary setup with only positives; it’s included for convenience.
> - Keep `temperature=0` for determinism across runs.

---

## 📦 Installation
```bash
pip install openai  # or your OpenAI-compatible client
```

If you use a compatible endpoint (e.g., self-hosted or proxy), provide `base_url` accordingly.

---

## ⚙️ Configuration
Set your credentials in environment variables or directly when constructing the class.

```python
from openai import OpenAI

EVAL = Evaluator(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    model="gpt-4o-mini",
)
```

---

## 🚀 Quick Start
```python
from openai import OpenAI

evaluator = Evaluator(api_key="...", base_url="...")

docs = [
    {"document": "The capital of France is Paris."},
    {"document": "Bananas are rich in potassium."}
]
query = "What is the capital of France?"

labels = evaluator.get_relevance_labels(docs, query, top_k=2)
metrics = evaluator.compute_metrics(show=True)
```

**Labeling prompt used (default):**
```text
Here are a question and a retrieved passage from a text corpus from the same domain as the question.

Can you judge whether an answer to the question can be derived from the retrieved passage, simply answer either “YES” or “NO”.

<binary>

Question: {query}; Retrieved Passage: {document}
```

---

## 🧠 Evaluating Embeddings, Rerankers, & Chunking
You can apply the evaluator across a test set and compare configurations.

### What can be evaluated?
- **Embedding model**: e.g., OpenAI vs. BGE
- **Top‑K (`top_k`)**: size of retrieved list (e.g., 20, 30)
- **Top‑N (`top_n`)**: docs forwarded to the generator (N ≤ K)
- **Semantic chunking threshold**: affects chunk size/coherence (typical: 1.5 → 3.0)
- **Reranking model**: e.g., FlashRank, Cohere, Jina Rank

### Minimal evaluation loop
```python
def evaluate_run(evaluator, dataset, top_k=10, ks=(1,3,5,10)):
    """dataset: list of {"query": str, "docs": list[{"document": str, ...}]}
    returns: list of per-query metrics dicts
    """
    run_results = []
    for example in dataset:
        evaluator.get_relevance_labels(example["docs"], example["query"], top_k=top_k)
        run_results.append(evaluator.compute_metrics(ks=list(ks), show=False))
    return run_results

from collections import defaultdict

def average_answer_presence(per_query_metrics):
    # returns {"@k": avg_answer_presence}
    acc = defaultdict(list)
    for m in per_query_metrics:
        for k, vals in m.items():
            acc[k].append(vals["Answer Presence"])  # 0/1
    return {k: sum(v)/len(v) for k, v in acc.items()}

# Example usage
per_query = evaluate_run(evaluator, dataset=my_test_set, top_k=20, ks=(1,2,3,5,10))
avg_ap = average_answer_presence(per_query)
print("Average Answer Presence:", avg_ap)
```

> **Tip:** Run the loop for each configuration (embedding, reranker, chunking threshold, etc.) and compare the averaged **Answer Presence@k** curves.

---

## 📊 Metrics
For each cutoff **k** using the latest labels (`1` = relevant, `0` = not):

- **Precision@k** = TP / k  
- **Recall@k** = TP / (# relevant in top‑K)  
- **Accuracy@k** = TP / k (same as precision here)  
- **Answer Presence@k** = `1` if TP ≥ 1 else `0`

The framework primarily emphasizes **Answer Presence@k**, a robust proxy for a RAG system’s ability to provide at least one answerable document.

---

## 📈 Results Templates
Paste your numbers below to track experiments.

### Reranker vs. No Reranker
| k | Answer Presence (No Reranker) | Answer Presence (Reranker) |
|---|-------------------------------:|----------------------------:|
| 1 |                                |                             |
| 3 |                                |                             |
| 5 |                                |                             |
|10 |                                |                             |

### Semantic Chunker Threshold
| Threshold | k | Answer Presence |
|----------:|---|----------------:|
| 1.5       | 5 |                 |
| 2.0       | 5 |                 |
| 2.5       | 5 |                 |
| 3.0       | 5 |                 |

### BGE vs OpenAI (t = 2)
| k | Answer Presence (BGE) | Answer Presence (OpenAI) |
|---|----------------------:|-------------------------:|
| 1 |                       |                          |
| 3 |                       |                          |
| 5 |                       |                          |
|10 |                       |                          |

### 4.1‑mini vs 4o‑mini
| k | Answer Presence (4.1‑mini) | Answer Presence (4o‑mini) |
|---|----------------------------:|---------------------------:|
| 1 |                             |                            |
| 3 |                             |                            |
| 5 |                             |                            |
|10 |                             |                            |

> Consider also plotting **Answer Presence vs k** for each configuration to visualize gains from reranking or better chunking.

---

## 🧯 Practical Notes & Pitfalls
- **Cost & speed**: Labeling is cheap/fast with small models (e.g., `gpt-4o-mini`), but still scales with dataset size × top‑k.
- **Determinism**: Keep `temperature=0`. Consider seeding sampling if your client supports it.
- **Token limits**: Ensure chunks fit comfortably; truncate or summarize if needed.
- **Prompt sensitivity**: You can pass a custom `prompt_template` to tune strictness.
- **Label noise**: LLMs can misjudge borderline cases; spot‑check a sample.

---

## 🔌 API Compatibility
- Works with any OpenAI‑compatible **Chat Completions** API.
- Set `base_url` to target non‑OpenAI providers (self‑hosted, proxies, etc.).

---

## 📜 License
MIT — do whatever you want, but attribution appreciated.

---

## 🙌 Acknowledgements
Inspired by practical needs in building reliable RAG pipelines where recall of at least one answerable chunk matters most.
The default prompt was found in: https://blog.ml6.eu/unsupervised-evaluation-of-semantic-retrieval-by-generating-relevance-judgments-with-an-llm-judge-ea244cc80908
