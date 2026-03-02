import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
INPUT_FILE = Path("work_fields.json")
OUTPUT_FILE = Path("correlation_matrix.json")
EXPECTED_FIELD_COUNT = 180
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# Signal fusion weights
W_SEMANTIC = 0.5
W_MORPHOLOGICAL = 0.25
W_NEIGHBORHOOD = 0.25

# Value 10 is reserved for self-correlation (diagonal), so we select
# the top 9 neighbors per field, ranked 9 (most similar) down to 1.
TOP_K = 9


def load_work_fields(path: Path) -> list[dict]:
    """Load and validate the work fields JSON file.

    Args:
        path: Path to the JSON file containing work field definitions.

    Returns:
        List of work field dicts, each with keys: code, nameDe, nameEn,
        correlationMatrixId.
    """
    with open(path, "r", encoding="utf-8") as f:
        fields = json.load(f)

    assert isinstance(fields, list), "Expected a JSON array"
    assert len(fields) == EXPECTED_FIELD_COUNT, (
        f"Expected {EXPECTED_FIELD_COUNT} fields, got {len(fields)}"
    )

    required_keys = {"code", "nameDe", "nameEn", "correlationMatrixId"}
    for i, field in enumerate(fields):
        missing = required_keys - field.keys()
        assert not missing, f"Field {i} missing keys: {missing}"
        # Verify correlationMatrixId format
        assert field["correlationMatrixId"].startswith("w_"), (
            f"Field {i}: unexpected correlationMatrixId format: {field['correlationMatrixId']}"
        )

    return fields


def print_data_summary(fields: list[dict]) -> None:
    """Print a summary of the loaded work fields for exploration.

    Args:
        fields: List of work field dicts as returned by load_work_fields().
    """
    print(f"Loaded {len(fields)} work fields.\n")

    print("Sample entries:")
    print("-" * 70)
    for field in fields[:5]:
        print(f"  code: {field['code']:12s}  "
              f"DE: {field['nameDe'][:30]:30s}  "
              f"EN: {field['nameEn'][:30]:30s}  "
              f"matrixId: {field['correlationMatrixId']}")
    print(f"  ... and {len(fields) - 5} more\n")

    # Check for unique codes and matrix IDs
    codes = [f["code"] for f in fields]
    matrix_ids = [f["correlationMatrixId"] for f in fields]
    print(f"Unique codes: {len(set(codes))} (expected {EXPECTED_FIELD_COUNT})")
    print(f"Unique matrix IDs: {len(set(matrix_ids))} (expected {EXPECTED_FIELD_COUNT})")

    assert len(set(codes)) == EXPECTED_FIELD_COUNT, "Duplicate codes found!"
    assert len(set(matrix_ids)) == EXPECTED_FIELD_COUNT, "Duplicate matrix IDs found!"
    print("\nAll codes and matrix IDs are unique.")


def compute_semantic_similarity(fields: list[dict]) -> np.ndarray:
    """
    Semantic similarity via multilingual sentence embeddings.

    Uses paraphrase-multilingual-MiniLM-L12-v2 which:
      - Handles both German and English text natively
      - Is trained on paraphrase detection (ideal for short-text similarity)
      - Produces 384-dimensional embeddings

    Input text format: "nameDe / nameEn" to leverage both languages,
    giving the model more semantic context than either language alone.

    Args:
        fields: List of work field dicts with 'nameDe' and 'nameEn' keys.

    Returns:
        Symmetric similarity matrix of shape (N, N) with cosine similarities
        in [-1, 1], where N = len(fields).
    """
    print(f"  Model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [f"{field['nameDe']} / {field['nameEn']}" for field in fields]

    # Example of what we're embedding
    print(f"  Sample input: \"{texts[0]}\"")
    print(f"  Encoding {len(texts)} texts...")

    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings)

    print(f"  Similarity matrix shape: {sim_matrix.shape}")
    print(f"  Value range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")

    # Sanity check for top 3 most similar pairs for 3 different fields
    print('\n Check top 3 most similar pairs for 3 sample fields:')
    for idx in [0, 4, 8]:
        scores = sim_matrix[idx].copy()
        scores[idx] = -1  # exclude self
        top_3_indices = np.argsort(scores)[-3:][::-1]
        field_name = f"{fields[idx]['nameEn']}"
        neighbors = ", ".join(
            f"{fields[j]['nameEn']} ({scores[j]:.3f})" for j in top_3_indices
        )
        print(f"    {field_name} → {neighbors}")
    return sim_matrix


def compute_morphological_similarity(fields: list[dict]) -> np.ndarray:
    """
    Morphological similarity via character n-gram TF-IDF.

    German compound nouns encode semantic relationships in their morphology.
    Character n-grams (3-5 chars) capture shared subword patterns that
    sentence embeddings may overlook:
      - "Personalentwicklung" & "Personalberatung" share "personal"
      - "Medizintechnik" & "Elektrotechnik" share "technik"
      - "Pharmazie" & "Pharmacy" share "pharma" (cross-language cognates)

    Word-level TF-IDF was tested but produced near-zero similarity
    because field names have too few unique words (2-6 each).
    Character n-grams overcome this by operating at sub-word granularity.

    Args:
        fields: List of work field dicts with 'nameDe' and 'nameEn' keys.

    Returns:
        Symmetric similarity matrix of shape (N, N) with cosine similarities
        in [0, 1], where N = len(fields).
    """
    print("\n\nComputing morphological similarity (character n-gram TF-IDF)")

    texts = [f"{f['nameDe']} {f['nameEn']}" for f in fields]
    print(f"  Sample input: \"{texts[0]}\"")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",       # character n-grams within word boundaries
        ngram_range=(3, 5),       # trigrams to 5-grams
        lowercase=True,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    print(f"  Vocabulary size (n-grams): {len(vectorizer.vocabulary_)}")
    print(f"  Matrix shape: {tfidf_matrix.shape}")

    sim_matrix = cosine_similarity(tfidf_matrix)

    print(f"  Value range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")

    print("\n  Sanity check for top 3 most similar for 3 sample fields:")
    for idx in [0, 4, 8]:
        scores = sim_matrix[idx].copy()
        scores[idx] = -1
        top3_indices = np.argsort(scores)[-3:][::-1]
        field_name = fields[idx]["nameEn"]
        neighbors = ", ".join(
            f"{fields[j]['nameEn']} ({scores[j]:.3f})" for j in top3_indices
        )
        print(f"    {field_name} → {neighbors}")

    print()
    return sim_matrix


def compute_neighborhood_similarity(
    fields: list[dict], sim_semantic: np.ndarray, k: int = 14
) -> np.ndarray:
    """
    Second-order (structural) similarity via k-NN neighborhood overlap.

    Instead of measuring direct similarity between two fields, this measures
    how much their local neighborhoods overlap. Two fields that share many
    nearest neighbors occupy the same region of the occupational landscape,
    even if their direct similarity is only moderate.

    Uses Jaccard similarity on the k-nearest-neighbor sets derived from
    the semantic similarity matrix.

    Args:
        fields: List of work field dicts.
        sim_semantic: Precomputed semantic similarity matrix of shape (N, N).
        k: Neighborhood size. Default 14 ≈ √180, a common heuristic for
            k-NN neighborhood size.

    Returns:
        Symmetric similarity matrix of shape (N, N) with Jaccard similarities
        in [0, 1], where N = len(fields).
    """
    print(f"Computing neighborhood overlap similarity (k={k})")

    n = len(fields)

    # Step 1: Find k nearest neighbors for each field
    neighborhoods: list[set[int]] = []
    for i in range(n):
        scores = sim_semantic[i].copy()
        scores[i] = -1  # exclude self
        top_k_indices = set(np.argsort(scores)[-k:])
        neighborhoods.append(top_k_indices)

    # Step 2: Compute pairwise Jaccard similarity on neighbor sets
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                intersection = len(neighborhoods[i] & neighborhoods[j])
                union = len(neighborhoods[i] | neighborhoods[j])
                jaccard = intersection / union if union > 0 else 0.0
                sim_matrix[i][j] = jaccard
                sim_matrix[j][i] = jaccard

    print(f"  Value range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")

    print("\n  Sanity check for top 3 most similar for 3 sample fields:")
    for idx in [0, 4, 8]:
        scores = sim_matrix[idx].copy()
        scores[idx] = -1
        top3_indices = np.argsort(scores)[-3:][::-1]
        field_name = fields[idx]["nameEn"]
        neighbors = ", ".join(
            f"{fields[j]['nameEn']} ({scores[j]:.3f})" for j in top3_indices
        )
        print(f"    {field_name} → {neighbors}")

    print()
    return sim_matrix


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Min-max normalize a similarity matrix to [0, 1].

    Each signal has a different native range (cosine: [-0.13, 1.0],
    Jaccard: [0, 1], char n-gram cosine: [0, 1]). Normalization
    ensures they contribute proportionally according to their weights.

    Args:
        matrix: A 2-D numpy similarity matrix.

    Returns:
        Normalized matrix with values in [0, 1]. Returns zeros if all
        values are identical (max == min).
    """
    min_val = matrix.min()
    max_val = matrix.max()
    if max_val == min_val:
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)


def build_correlation_entries(
    fields: list[dict], sim_combined: np.ndarray
) -> list[dict]:
    """
    Build the output entries from the fused similarity matrix.

    For each field:
      1. Find the top 9 most similar fields (excluding self)
      2. Assign integer rank: most similar = 9, 9th = 1
        (value 10 is reserved for self-correlation on the diagonal)
      3. Store in upper triangle form (code1 < code2 lexicographically)

    When a pair appears from both sides (A's top-9 includes B, and
    B's top-9 includes A), we keep the higher rank value to avoid
    losing strong relationships.

    Args:
        fields: List of work field dicts with 'correlationMatrixId' key.
        sim_combined: Fused similarity matrix of shape (N, N).

    Returns:
        List of entry dicts, each with keys: code1, code2, value (int 1-10),
        score (float). Includes N diagonal entries (value=10) and up to
        N * TOP_K non-diagonal entries (after deduplication).
    """
    n = len(fields)
    matrix_ids = [f["correlationMatrixId"] for f in fields]
    entries = []

    # Diagonal: self-correlation = 10
    for i in range(n):
        entries.append({
            "code1": matrix_ids[i],
            "code2": matrix_ids[i],
            "value": 10,
            "score": 1.0,
        })

    # For each field, find top 10 most similar
    pair_values: dict[tuple[str, str], dict] = {}

    for i in range(n):
        scores = sim_combined[i].copy()
        scores[i] = -1  # exclude self
        top_indices = np.argsort(scores)[-TOP_K:][::-1]

        for rank, j in enumerate(top_indices):
            value = TOP_K - rank  # 9, 8, 7, ..., 1

            # Upper triangle: code1 < code2
            pair = tuple(sorted([matrix_ids[i], matrix_ids[j]]))

            # Keep the higher value when pair appears from both sides
            if pair not in pair_values or value > pair_values[pair]["value"]:
                pair_values[pair] = {
                    "code1": pair[0],
                    "code2": pair[1],
                    "value": value,
                    "score": round(float(scores[j]), 4),
                }

    entries.extend(pair_values.values())
    return entries


def validate_output(entries: list[dict], field_count: int) -> None:
    """
    Verify the output satisfies all four required matrix properties:
      1. Symmetry: upper triangle only, no duplicate pairs
      2. Sparsity: only top 10 per field included (satisfied by construction in build_correlation_entries function)
      3. Diagonal: self-correlation present for every field, value = 10
      4. Value range: integer ranks from 1 to 10

    Args:
        entries: List of correlation entry dicts as returned by
            build_correlation_entries().
        field_count: Expected number of distinct work fields (for diagonal
            entry count validation).

    Raises:
        AssertionError: If any matrix property is violated.
    """
    print("Validating output:")

    # Property 1: Symmetry — upper triangle only, no duplicates
    non_diag = [e for e in entries if e["code1"] != e["code2"]]
    for e in non_diag:
        assert e["code1"] < e["code2"], (
            f"Not upper triangle: {e['code1']} >= {e['code2']}"
        )
    pairs = [(e["code1"], e["code2"]) for e in non_diag]
    assert len(pairs) == len(set(pairs)), "Duplicate pairs found!"
    print(f"  Non-diagonal entries: {len(non_diag)} (all upper triangle, no duplicates)")

    # Property 3: Diagonal
    diag = [e for e in entries if e["code1"] == e["code2"]]
    assert len(diag) == field_count, (
        f"Expected {field_count} diagonal entries, got {len(diag)}"
    )
    assert all(e["value"] == 10 for e in diag), "All diagonal values must be 10"
    print(f"  Diagonal entries: {len(diag)} (all value=10)")

    # Property 4: Value range
    for e in entries:
        assert isinstance(e["value"], int), f"Value must be int: {e['value']}"
        assert 1 <= e["value"] <= 10, f"Value out of range: {e['value']}"
    print("  All values are integers in [1, 10]")

    print(f"  Total entries: {len(entries)}")
    print("  All validations passed.\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("Work Field Correlation Matrix Generator")
    print("=" * 70 + "\n")

    # Step 1: Load and validate data
    fields = load_work_fields(INPUT_FILE)
    print_data_summary(fields)

    # Step 2: Compute similarity signals
    print("\n" + "=" * 70)
    print("Computing similarity signals")
    print("=" * 70 + "\n")

    sim_semantic = compute_semantic_similarity(fields)
    sim_morphological = compute_morphological_similarity(fields)
    sim_neighborhood = compute_neighborhood_similarity(fields, sim_semantic)

    # Step 3: Fuse signals
    print("=" * 70)
    print("Fusing similarity signals")
    print("=" * 70 + "\n")
    print(f"  Weights: semantic={W_SEMANTIC}, morphological={W_MORPHOLOGICAL}, "
          f"neighborhood={W_NEIGHBORHOOD}")

    sim_combined = (
        W_SEMANTIC * normalize_matrix(sim_semantic)
        + W_MORPHOLOGICAL * normalize_matrix(sim_morphological)
        + W_NEIGHBORHOOD * normalize_matrix(sim_neighborhood)
    )
    print(f"  Combined matrix range: [{sim_combined.min():.4f}, {sim_combined.max():.4f}]")

    # Spot check fused results
    print("\n  Check top 5 most similar for sample fields (fused):")
    for idx in [0, 4, 8]:
        scores = sim_combined[idx].copy()
        scores[idx] = -1
        top5 = np.argsort(scores)[-5:][::-1]
        field_name = fields[idx]["nameEn"]
        neighbors = ", ".join(
            f"{fields[j]['nameEn']} ({scores[j]:.3f})" for j in top5
        )
        print(f"    {field_name} → {neighbors}")
    print()

    # Step 4: Build ranked entries
    print("=" * 70)
    print("Building correlation entries")
    print("=" * 70 + "\n")

    entries = build_correlation_entries(fields, sim_combined)

    # Step 5: Validate
    validate_output(entries, EXPECTED_FIELD_COUNT)

    # Step 6: Export
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Output written to {OUTPUT_FILE}")
    print(f"Total entries: {len(entries)}")


if __name__ == "__main__":
    main()
