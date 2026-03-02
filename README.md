# Work Field Correlation Matrix

## Overview

Generates a meaningful correlation matrix over 180 occupational work fields.
Rather than relying on a single similarity measure, the pipeline combines three
complementary signals: semantic embeddings, morphological character n-grams,
and structural neighborhood overlap, into a fused similarity score, then ranks
the top neighbors for each field into integer values from 1 to 10.

Input: `work_fields.json` (180 entries with German and English names).
Output: `correlation_matrix.json` (sparse upper-triangle matrix with ranked correlations).

## Quick Start

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python generate_matrix.py
```

The script prints progress and sanity checks to stdout, then writes
`correlation_matrix.json` to the working directory.

## Methodology

The pipeline computes three independent similarity signals, which capture meaning, form, and context, and normalizes each to
[0, 1], and fuses them with fixed weights into a single combined score.

### Signal 1: Semantic Similarity

Model: `paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional sentence embeddings).

This model was chosen because:

- It handles both German and English text natively, requiring no translation step.
- It is trained on paraphrase detection, which is well-suited for matching short
  occupational labels that describe the same concept in different words.

Each field is encoded as `"nameDe / nameEn"` (e.g., `"Telekommunikation / Telecomunication"`),
giving the model bilingual context. Pairwise cosine similarity produces a 180x180 matrix.

Weight in fusion: **0.50**

### Signal 2: Morphological Similarity

Method: Character n-gram TF-IDF (trigrams to 5-grams, within word boundaries).

German compound nouns encode semantic relationships directly in their morphology.
Character n-grams capture shared subword patterns that sentence embeddings may miss:

- "Personalentwicklung" and "Personalberatung" share the stem "personal"
- "Medizintechnik" and "Elektrotechnik" share "technik"
- "Pharmazie" and "Pharmacy" share "pharma" (cross-language cognate)

Word-level TF-IDF was tested first but produced near-zero similarity because
field names contain only 2-6 unique words each -- far too few for meaningful
term frequency statistics. Character n-grams yield much better results by operating at
sub-word granularity, finding thousands of overlapping features per field name.

Weight in fusion: **0.25**

### Signal 3: Neighborhood Overlap

Method: Jaccard similarity on k-nearest-neighbor sets (k=14, approximately sqrt(180)).

Instead of measuring direct similarity between two fields, this captures
second-order structure: two fields that share many nearest neighbors in the
semantic embedding space occupy the same region of the occupational landscape,
even if their direct similarity is only moderate.

For each field, the 14 nearest neighbors (by semantic similarity) are identified.
Then for every pair of fields, the Jaccard index of their neighbor sets is computed.
This produces a similarity signal that is structurally distinct from the direct
cosine similarity it is derived from.

Weight in fusion: **0.25**

### Signal Fusion

Each of the three raw similarity matrices has a different native value range
(semantic cosine can be negative, Jaccard is in [0, 1], character n-gram cosine
is in [0, 1]). Before combining, each matrix is min-max normalized to [0, 1].

The fused score for each pair is:

```python
combined = 0.50 * semantic_norm + 0.25 * morphological_norm + 0.25 * neighborhood_norm
```

Semantic similarity receives the highest weight because it captures the broadest
notion of meaning. The other two signals each contribute a distinct perspective --
surface-level morphology and structural neighborhood position -- at equal weight.

### Ranking and Output

For each of the 180 fields, the top 9 most similar fields (by fused score) are
selected. These are assigned integer ranks: the most similar neighbor receives
value 9, the second-most similar receives 8, and so on down to 1.

Value 10 is reserved exclusively for the diagonal (self-correlation).

When a pair appears from both sides (field A has B in its top-9, and field B
also has A in its top-9), the higher rank value is kept. This avoids losing
strong bidirectional relationships.

All entries are stored in upper-triangle form (code1 < code2 lexicographically).

## Design Decisions

Alternatives that were explored and why they were set aside:

- **Word-level TF-IDF**: Produced near-zero similarity across all pairs. Field names
  have only 2-6 unique words each, which is too few for meaningful term frequency
  statistics. Replaced by character n-gram TF-IDF which operates at sub-word level.

- **LLM-based scoring**: Would produce high-quality pairings but is inherently
  non-deterministic (different runs or API versions yield different scores).
  Reproducibility was a hard requirement.

- **Hand-crafted taxonomy rules**: Keyword-based category assignment (e.g., fields
  containing "tech" go into an Engineering group). Fragile, requires too much manual categorization rather than algorithmic approach for
  180 diverse fields, and difficult to justify the rules objectively.

- **Cluster-based similarity**: Agglomerative clustering was tested to derive a
  third signal (same-cluster bonus). Silhouette analysis showed monotonically
  increasing scores from 0.06 to 0.11 across cluster counts 5-30, which indicated
  no natural cluster structure in the data. Replaced by neighborhood overlap
  which captures structural similarity without assuming discrete groups.

## Reproducibility

The entire pipeline is deterministic. Running `python generate_matrix.py` on
the same input will always produce the same output:

- Sentence embeddings are computed with a fixed model and no random sampling.
- TF-IDF vectorization is deterministic by construction.
- Jaccard neighborhood overlap is computed from fixed neighbor sets.
- No random seeds are needed because no stochastic steps exist in the pipeline.

Dependency versions are pinned in `requirements.txt` to prevent changes in
library behavior across environments.
