import h5py
from pathlib import Path

# Define the path to your H5 file
H5_FILE_PATH = Path("./03_morphometrics_output_combined/original_pca_scores_and_class_labels_combined.h5")

def inspect_h5_file(file_path: Path):
    """
    Opens an H5 file and prints the names of all top-level datasets and groups.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Contents of {file_path}:")
            print("-" * 30)
            if not f.keys():
                print("No datasets or groups found in this file.")
            for key in f.keys():
                item = f[key]
                print(f"  - {key} (Type: {type(item).__name__}, Shape: {item.shape if hasattr(item, 'shape') else 'N/A'})")
            print("-" * 30)
    except Exception as e:
        print(f"An error occurred while opening or reading the H5 file: {e}")

if __name__ == "__main__":
    inspect_h5_file(H5_FILE_PATH)