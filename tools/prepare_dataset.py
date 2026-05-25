"""Dataset preparation helpers.

Converts dataset-specific formats into the clean A/B/Mask structure.
Edit this file to add new dataset converters.
Usage: python tools/prepare_dataset.py
"""


def main() -> None:
    print("Prepare datasets into this structure:")
    print("  datasets/<DATASET>/<split>/A")
    print("  datasets/<DATASET>/<split>/B")
    print("  datasets/<DATASET>/<split>/Mask")
    print("Supported splits are configured in configs/active.yaml.")


if __name__ == "__main__":
    main()
