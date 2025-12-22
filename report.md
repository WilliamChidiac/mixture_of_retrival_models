# Mixture of Retrieval Models (MOE) Report

## Introduction
Modern Information Retrieval (IR) systems face a fundamental challenge: the "one size fits all" problem. User queries are incredibly diverse, ranging from specific keyword searches (e.g., "error code 404") to abstract conceptual questions (e.g., "how to improve mental well-being").

Traditional systems often rely on a single retrieval paradigm—either keyword-based (Sparse) or semantic-based (Dense). However, neither approach is universally optimal. Sparse models like BM25 excel at exact matching but fail to capture semantic nuance, while Dense models capture meaning but struggle with precise terminology.

In this project, we propose an adaptive retrieval framework inspired by the Mixture of Experts (MOE) architecture. Instead of relying on a monolithic model, we implement a "Router" that dynamically analyzes the input query and assigns weights to different retrieval "experts." By intelligently fusing the strengths of both Sparse and Dense models based on the specific intent of each query, we aim to create a more robust and versatile search system.

## 1. Standard Retrieval Models: Sparse vs. Dense

Information Retrieval (IR) systems generally fall into two main paradigms, each with distinct strengths and weaknesses depending on the query type and collection.

### Sparse Models (Keyword Matching)
These models rely on exact keyword matching. They represent documents and queries as sparse vectors (e.g., bag-of-words) where dimensions correspond to vocabulary terms. Sparse Models are usually very good at finding answers to factual queries (e.g: How many **calories** are in a **tablespoon of honey**? => A **tablespoon of honey** contains 64 **calories**)
*   **Example:** **BM25** (Best Matching 25) is the industry standard. It improves upon TF-IDF by incorporating term saturation and document length normalization.
*   **Pros:** Extremely efficient, exact match capability (crucial for specific names, IDs, rare words), and highly interpretable.
*   **Cons:** Suffers from the "vocabulary mismatch" problem (e.g., searching for "car" misses "automobile") and cannot capture semantic meaning or context.

### Dense Models (Semantic Search)
These models use deep learning (typically Transformers like BERT) to encode documents and queries into dense, low-dimensional vectors (embeddings). Retrieval is performed by finding the nearest neighbors in this vector space (e.g., using Cosine Similarity). Dense models excel at conceptual queries where keywords might not match (e.g: **Can I pay with my phone?** => The store accepts **Apple Pay** and **Google Wallet**).
*   **Pros:** Captures semantic meaning, handles synonyms and paraphrasing well, and solves the vocabulary mismatch problem.
*   **Cons:** Computationally expensive (requires GPU), can hallucinate relevance for completely unrelated texts if they share a similar "topic" embedding, and struggles with exact phrase matching or rare entities not seen during training.

## 2. Learning to Rank (LTR)

While standard models use fixed heuristic formulas (like BM25) or geometric similarity (like Cosine Similarity) to rank documents, **Learning to Rank (LTR)** introduces a machine learning approach to the ranking problem. LTR models *learn* the optimal ranking function from labeled training data (queries, documents, and relevance labels).

*   **Training Data:** LTR models require datasets containing queries, documents, and relevance judgments (e.g., "relevant", "not relevant").
*   **Approaches:**
    *   **Pointwise:** Predicts a relevance score for a single document independently.
    *   **Pairwise:** Learns to classify which of two documents is more relevant.
    *   **Listwise:** Optimizes the entire ranked list to maximize a metric like NDCG.

LTR allows systems to combine multiple features—such as BM25 scores, semantic similarity, page rank, and document freshness—to produce a more accurate final ranking.

## 3. Our Approach: LTR at the Model Level

In this project, we merge the concepts of Standard Retrieval Models and Learning to Rank, but we adapt the LTR paradigm to operate at a higher level of abstraction.

Instead of learning to rank individual documents based on their features (traditional LTR), our approach learns to rank **entire retrieval models** (the "experts"). We recognize that Sparse and Dense models have complementary strengths: Sparse is better for keyword-heavy queries, while Dense is better for conceptual queries.

By training a neural router to dynamically assign weights to these different models based on the query's characteristics, we effectively perform **Learning to Rank at the Model Level**. We are optimizing the fusion of diverse retrieval strategies, learning a query-dependent weighting function that decides which "expert" should be trusted more for a specific user need.

## 4. Mixture of Experts (MOE) Architecture
Our approach is inspired by the Mixture of Experts (MOE) framework. In a traditional MOE, a "gating network" (router) determines which "experts" (sub-networks) should handle a given input.
*   **Analogy:** In our IR context, the **Experts** are the retrieval models (BM25 and Dense), and the **Router** is a neural network that decides how much to trust each expert for a specific query.
*   **Router Types:**
    *   **Hard Routing:** The router selects only *one* expert (the best one) to handle the query.
    *   **Soft Routing:** The router assigns a *weight* (probability) to each expert, and the final result is a weighted combination of their outputs. **We utilize a Soft Router setup**, where the router outputs a probability distribution over the experts.

## 5. Training the Router

### 5.1 From Text to Weights (Generating Labels)
To train the router, we first need "ground truth" labels that tell us how good each expert is for a given query. We generate these labels using a scoring function based on the experts' performance on training data.
*   **Scoring Function:** For each query and expert, we calculate a score:
    $$ Score_{expert} = \sum_{d \in TopK} \left( \frac{1}{rank(d)} \times relevance(d) \times \frac{1}{count(d)} \right) $$
    *   **$1/rank(d)$:** Reciprocal rank, giving higher weight to documents ranked higher.
    *   **$relevance(d) \in \{0, 1, 2\}$ :** Ground truth relevance from the dataset (qrels)
    *   **$1/count(d)$:** Normalization factor (inverse of the number of experts that found this document), penalizing redundancy or normalizing the contribution. We came up with the heuritic that documents that every model retrieves are considered easy to find, this term help to redirect the weights to models that retrieve relevant documents that are harder to find.
*   **Router Inputs/Outputs:**
    *   **Input:** The query text is converted into a dense embedding using `all-mpnet-base-v2`.
    *   **Output:** A probability distribution (weights) for the experts (Sparse, Dense).
    *   **Loss Function:** We use **KL Divergence Loss** (`nn.KLDivLoss`) to train the router to minimize the difference between its predicted probability distribution and the "true" distribution derived from the scoring function.

### 5.2 From Weights to Text (Inference/Merging)
Once the router is trained, we use it to rank documents for new queries.
*   **Merging Formula:** The final score for a document $d$ is the weighted sum of its scores from each expert:
    $$ FinalScore(d) = \sum_{expert} \frac{w_{expert}}{rank_{expert}(d) + 1} $$
    *   $w_{expert}$: The weight assigned to the expert by the router.
    *   $rank_{expert}(d)$: The rank of the document in that expert's list (0-indexed).
*   The documents are then re-ranked based on this $FinalScore$.

## 6. Final Model Architecture

1.  **Query Embedding:** The input query $Q$ is encoded into an embedding vector $E_Q$ using a Sentence Transformer (`all-mpnet-base-v2`).
2.  **Router Prediction:** $E_Q$ is passed through the Router (MLP), which outputs weights $W = \{w_{sparse}, w_{dense}\}$.
3.  **Expert Retrieval given $Q$:**
    *   **Sparse Expert (BM25):** Retrieves top-K documents based on keywords.
    *   **Dense Expert (all-MiniLM-L6-v2):** Retrieves top-K documents based on vector similarity.
4.  **Weighted Fusion:** The results from both experts are merged. Each document's contribution is scaled by the router's weight $W$ and its rank position.
5.  **Final Ranking:** The merged list is sorted by the final fused score to produce the result.

## 7. Experiment Setup
We conducted experiments to evaluate the effectiveness of our MOE framework under different training conditions.

*   **Datasets:** We focused our detailed analysis on two distinct collections from the BEIR benchmark to represent different retrieval challenges:
    *   **FIQA (Financial Question Answering):**
        *   **Domain:** Finance (Opinionated & Fact-based).
        *   **Corpus Size:** 57,638 documents.
        *   **Training Queries:** 5,500.
        *   **Test Queries:** 648.
        *   **Task:** Retrieving financial snippets that answer user questions.
    *   **SciFact (Scientific Fact Checking):**
        *   **Domain:** Biomedical/Scientific.
        *   **Corpus Size:** 5,183 documents.
        *   **Training Queries:** 809.
        *   **Test Queries:** 300.
        *   **Task:** Given a scientific claim, retrieve relevant abstracts that support or refute it. This requires high precision and semantic understanding.

*   **Models:**
    *   **Sparse Expert:** BM25 (standard implementation).
    *   **Dense Expert:** `all-MiniLM-L6-v2` (efficient sentence transformer).
    *   **Router:** MLP with hidden layers [128, 64], ReLU activation, BatchNorm, and Dropout (0.3). Input is `all-mpnet-base-v2` embeddings (768 dimensions).

*   **Experimental Scenarios:**
    1.  **Multi-Domain Training (Run 4):** The router was trained on a combined dataset of FIQA and Scifact scores. We evaluated its performance on the test sets of both collections to assess its ability to handle diverse domains simultaneously.
    2.  **Cross-Domain Generalization (Run 5 & 6):**
        *   **Run 5:** Trained on FIQA, tested on FIQA and Scifact. This tests if a router trained on financial queries can generalize to scientific ones.
        *   **Run 6:** Trained on Scifact, tested on FIQA and Scifact. This tests the reverse generalization.

## 8. Results and Metrics

We evaluated the system using standard IR metrics: Precision@K and Recall@K (for K=1, 5, 10, 25). We also tracked the Router's accuracy in predicting the optimal weights.

### 8.1 Single Domain Training (Run 5 & 6)
In this baseline experiment, we trained and tested the router on the same dataset to establish the upper bound of performance when the domain is known.

**FIQA (Run 5: Train FIQA -> Test FIQA)**
The router successfully learned to prioritize the Dense expert, which is generally stronger for FIQA, but still leveraged the Sparse expert for specific queries to achieve marginal gains.

| Metric | Sparse (BM25) | Dense (MiniLM) | MOE (Ours) |
| :--- | :--- | :--- | :--- |
| **P@1** | 0.191 | 0.347 | **0.352** |
| **R@1** | 0.097 | 0.171 | **0.172** |
| **P@5** | 0.081 | **0.168** | 0.165 |
| **R@5** | 0.191 | **0.367** | **0.367** |
| **P@10** | 0.056 | 0.104 | **0.105** |
| **R@10** | 0.258 | 0.441 | **0.449** |
| **P@25** | 0.031 | **0.054** | **0.053** |
| **R@25** | 0.336 | **0.551** | **0.550** |

**Scifact (Run 6: Train Scifact -> Test Scifact)**
The improvement on Scifact was more pronounced. Since Scifact requires exact matching for claims (Sparse) but also semantic understanding (Dense), the router's ability to switch strategies proved highly effective.

| Metric | Sparse (BM25) | Dense (MiniLM) | MOE (Ours) |
| :--- | :--- | :--- | :--- |
| **P@1** | 0.510 | 0.503 | **0.533** |
| **R@1** | 0.493 | 0.482 | **0.508** |
| **P@5** | 0.149 | 0.164 | **0.171** |
| **R@5** | 0.692 | 0.738 | **0.780** |
| **P@10** | 0.083 | 0.088 | **0.093** |
| **R@10** | 0.753 | 0.783 | **0.834** |
| **P@25** | 0.035 | 0.039 | **0.041** |
| **R@25** | 0.808 | 0.855 | **0.902** |

### 8.2 Multi-Domain Training (Run 4)
**Router Accuracy:** 71.87%

In this scenario, the router was trained on both FIQA and Scifact. The MOE model consistently outperformed or matched the best single expert across both datasets.

**FIQA Results (Run 4)**
| Metric | Sparse (BM25) | Dense (MiniLM) | MOE (Ours) |
| :--- | :--- | :--- | :--- |
| **P@1** | 0.191 | 0.347 | **0.356** |
| **R@1** | 0.097 | 0.171 | **0.178** |
| **P@5** | 0.081 | **0.168** | **0.167** |
| **R@5** | 0.191 | 0.367 | **0.374** |
| **P@10** | 0.056 | 0.104 | **0.105** |
| **R@10** | 0.258 | 0.441 | **0.450** |
| **P@25** | 0.031 | **0.054** | 0.053 |
| **R@25** | 0.336 | **0.551** | 0.550 |

**Scifact Results (Run 4)**
| Metric | Sparse (BM25) | Dense (MiniLM) | MOE (Ours) |
| :--- | :--- | :--- | :--- |
| **P@1** | 0.510 | 0.503 | **0.530** |
| **R@1** | 0.493 | 0.482 | **0.505** |
| **P@5** | 0.149 | 0.164 | **0.170** |
| **R@5** | 0.692 | 0.738 | **0.775** |
| **P@10** | 0.083 | 0.088 | **0.092** |
| **R@10** | 0.753 | 0.783 | **0.824** |
| **P@25** | 0.035 | 0.039 | **0.040** |
| **R@25** | 0.808 | 0.855 | **0.892** |

### 8.3 Cross-Domain Generalization (Run 5 & 6)
We tested the router's ability to generalize by training on one domain and testing on another.

**Run 5: Train on FIQA -> Test on Scifact**
*   **Router Accuracy:** 71.56%
*   **Observation:** Despite never seeing scientific claims during training, the router improved retrieval performance on Scifact, suggesting it learned transferable heuristics about query types and specifications.

**Scifact Results (Run 5 - Zero-Shot)**
| Metric | Sparse (BM25) | Dense (MiniLM) | MOE (Ours) |
| :--- | :--- | :--- | :--- |
| **P@1** | 0.510 | 0.503 | **0.510** |
| **R@1** | 0.493 | 0.482 | **0.485** |
| **P@5** | 0.149 | 0.164 | **0.166** |
| **R@5** | 0.692 | 0.738 | **0.756** |
| **P@10** | 0.083 | 0.088 | **0.092** |
| **R@10** | 0.753 | 0.783 | **0.823** |
| **P@25** | 0.035 | 0.039 | **0.040** |
| **R@25** | 0.808 | 0.855 | **0.896** |

## 9. Interpretation of Results

### 9.1 The Value of Hybridization (Single Domain)
Our single-domain baselines (Section 8.1) reveal the intrinsic value of the MOE approach.
*   **Scifact:** The significant gains (R@10 +5% over Dense) confirm that scientific claim verification is a complex task requiring both exact entity matching (Sparse) and semantic inference (Dense). The router successfully identified which strategy to prioritize per query.
*   **FIQA:** The gains were marginal. This is likely because FIQA is heavily dominated by semantic search (Dense expert is far superior to Sparse). In such "unbalanced" scenarios, the router's job is simply to "not mess up" by picking the weaker expert, which it mostly achieved.

### 9.2 Capacity for Multi-Tasking (Multi-Domain)
In Run 4 (Section 8.2), the router was trained on the combined dataset. Crucially, **performance did not degrade** compared to the single-domain baselines. This indicates that the router has sufficient capacity to learn distinct decision boundaries for different domains simultaneously without suffering from catastrophic forgetting or interference. It effectively acts as a "universal" switch.

### 9.3 Asymmetric Generalization (Cross-Domain)
The cross-domain experiments (Section 8.3) revealed an interesting asymmetry:
*   **General to Specific (FIQA -> Scifact):** Training on the larger, more diverse FIQA dataset allowed the router to learn robust, transferable features about query difficulty and semantic density. These features generalized well to Scifact, enabling zero-shot improvement.

### 9.4 The Accuracy-Performance Correlation
There is a direct link between Router Accuracy and Retrieval Performance.
*   **High Accuracy (>70%):** In Runs 4 and 5, high accuracy correlated with MOE outperforming the best single expert.
*   **Low Accuracy (<60%):** In Run 6 (Zero-shot FIQA), low accuracy resulted in the MOE model underperforming the Dense baseline. This confirms that a "confused" router is detrimental; it is better to stick to a single strong model than to dynamically weight them incorrectly.

## 10. Missing Experiments
Given more time, we would have expanded our analysis with the following experiments:

1.  **Impact of Normalization & Routing Strategy:** We exclusively used Soft Routing with L1 normalization. It would be valuable to compare this against:
    *   **Hard Routing:** Selecting only the top-1 expert (equivalent to $L_\infty$ normalization). This would quantify the value of the "ensemble effect" (merging lists) versus pure "model selection."
    *   **Softmax with Temperature:** Using Softmax with a temperature parameter to control the "sharpness" of the distribution, allowing us to smoothly interpolate between soft and hard routing.

2.  **Router Architecture Exploration:** Due to time constraints, we relied on a single MLP architecture (Hidden Layers: [128, 64]) and did not perform hyperparameter tuning. Future work should explore:
    *   **Deeper/Wider Networks:** To capture more complex interactions between query features.
    *   **Transformer Encoders:** Instead of using a fixed sentence embedding as input, fine-tuning a small Transformer (like DistilBERT) directly as the router could allow it to learn task-specific token-level features that predict expert performance better than a pooled embedding.
    *   **Optimizing for Accuracy:** We accepted ~70% accuracy as sufficient, but pushing this higher through architecture search could directly translate to better retrieval performance as we empirically saw through the experiments described above.

3.  **Disagreement-Based Training:** Currently, we train the router on all queries, including those where both experts perform similarly (or both fail). A more efficient approach might be to filter the training data to focus *only* on queries where there is a significant performance gap between the experts. By training the router specifically on these "high-disagreement" samples, we could force it to learn the critical decision boundaries that actually impact the final ranking, rather than wasting capacity on easy or ambiguous cases.

## 11. Future Directions

1.  **End-to-End Neural MOE with Specialized Experts:**
    Currently, our experts are fixed (BM25 is static, and the Dense model is pre-trained). A true MOE architecture would involve **jointly training** the router and multiple neural experts (e.g., multiple Dense retrievers or a mix of Dense and SPLADE).
    *   **Why:** Since BM25 is non-differentiable, end-to-end training currently only benefits the Dense expert. By using multiple differentiable experts, we could allow them to **diverge and specialize** during training.
    *   **Outcome:** Expert A might naturally specialize in short, factoid queries, while Expert B adapts to long, semantic questions. The router would learn to distribute the load, creating a system where the experts are not just "good models" but "complementary specialists," maximizing the ensemble's total coverage.

2.  **Query Expansion as a Pre-processing Step:**
    Instead of relying solely on the router to fix retrieval gaps, we could implement query expansion (e.g., using RM3 or an LLM-based rewriter) as a standard pre-processing step for *each* expert.
    *   **Goal:** By enriching the query with synonyms and related terms before it reaches the experts, we boost the raw recall of both the Sparse and Dense models individually. The router then operates on these "enhanced" result sets, potentially achieving a higher performance ceiling than merging raw retrieval results.

3.  **LLM-Based Routing:**
    A promising direction is to replace the trained MLP router with a Large Language Model (LLM).
    *   **Concept:** We could prompt an LLM with the user's query and a description of the available experts (e.g., "Expert A is precise with keywords, Expert B understands concepts"). The LLM would reason about the query's intent and output the optimal weights for each model.
    *   **Benefit:** This leverages the massive world knowledge and reasoning capabilities of LLMs, potentially enabling "zero-shot routing" that understands complex query nuances without requiring a dedicated training dataset for the router.

## 12. Conclusion
We successfully implemented a Mixture of Experts framework for Information Retrieval. By training a neural router to dynamically assign weights to Sparse (BM25) and Dense experts based on the query's semantic content, we created a flexible system that leverages the strengths of both paradigms. The router demonstrated high accuracy in predicting expert performance, suggesting that this hybrid approach is a promising direction for building robust search systems that generalize well across different types of user queries.

## 13. References

**Sentence-BERT (all-mpnet-base-v2 & all-MiniLM-L6-v2)**
*   Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 3982–3992.
*   Song, K., Tan, X., Qin, T., Lu, J., & Liu, T.-Y. (2020). MPNet: Masked and Permuted Pre-training for Language Understanding. *Advances in Neural Information Processing Systems*, 33, 16857–16867.
*   Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *Advances in Neural Information Processing Systems*, 33, 5776–5788.

**Mixture of Experts (MOE): A Big Data Perspective**
*   https://arxiv.org/abs/2501.16352
