import json
from pathlib import Path

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
INPUT_FILE = Path("work_fields.json")
OUTPUT_FILE = Path("correlation_matrix.json")
EXPECTED_FIELD_COUNT = 180


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
    print("\n All codes and matrix IDs are unique.")


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


if __name__ == "__main__":
    main()
