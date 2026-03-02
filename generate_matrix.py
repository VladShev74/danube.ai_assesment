import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
INPUT_FILE = Path("work_fields.json")
OUTPUT_FILE = Path("correlation_matrix.json")
EXPECTED_FIELD_COUNT = 180
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_work_fields(path: Path) -> list[dict]:
    """Load and validate the work fields JSON file."""
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
    """Print a summary of the loaded work fields for exploration."""
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


def compute_semantic_similarity(fields):
    """
    Semantic similarity via multilingual sentence embeddings.

    Uses paraphrase-multilingual-MiniLM-L12-v2 which:
      - Handles both German and English text natively
      - Is trained on paraphrase detection (ideal for short-text similarity)
      - Produces 384-dimensional embeddings

    Input text format: "nameDe / nameEn" to leverage both languages,
    giving the model more semantic context than either language alone.
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
    print('\n Check top 3 most similar pairs for 3 different fields:')
    for idx in [0, 4, 8]:
        scores = sim_matrix[idx].copy()
        scores[idx] = -1  # exclude self
        top_3_indices = np.argsort(scores)[-3:][::-1]
        field_name = f"{fields[idx]['nameEn']}"
        neighbors = ", ".join(
            f"{fields[j]['nameEn']} ({scores[j]:.3f})" for j in top_3_indices
        )
        print(f"    {field_name} → {neighbors}")
    return None


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

    compute_semantic_similarity(fields)


if __name__ == "__main__":
    main()
