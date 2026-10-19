"""
Export and Packaging Module for Chemistry Multi-Agent Outputs
Collects generated questions (.json, .html, .docx, .pkl) and images,
packages them into a clean zip archive for download and submission.
"""

import os
import shutil
import argparse
from config import OUTPUT_DIR, ZIP_OUTPUT_PATH


def package_outputs(output_dir=OUTPUT_DIR, zip_base_name=ZIP_OUTPUT_PATH):
    """
    Gathers all generated question files and image folders into a zipped bundle.
    """
    print("=" * 60)
    print("PACKAGING GENERATED OUTPUTS")
    print("=" * 60)

    staging_dir = os.path.join(output_dir, "staging_outputs")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    target_extensions = (
        ".pkl", ".pkp", ".docx", ".doc", ".html", ".htm", ".json", ".csv", ".txt"
    )
    allowed_folders = ["generated_question_images", "chemistry_images"]

    copied_files = 0
    copied_folders = 0

    print("Step 1: Collecting output files and image folders...")
    for item in os.listdir(output_dir):
        src_path = os.path.join(output_dir, item)

        # Copy target files
        if os.path.isfile(src_path) and item.lower().endswith(target_extensions):
            shutil.copy2(src_path, staging_dir)
            copied_files += 1

        # Copy allowed image folders
        elif os.path.isdir(src_path) and item in allowed_folders:
            dest_folder = os.path.join(staging_dir, item)
            shutil.copytree(src_path, dest_folder)
            copied_folders += 1

    print(f"Staged {copied_files} files and {copied_folders} folders.")

    print("\nStep 2: Creating zip archive...")
    zip_file = f"{zip_base_name}.zip"
    if os.path.exists(zip_file):
        os.remove(zip_file)

    shutil.make_archive(zip_base_name, "zip", staging_dir)

    # Clean up staging directory
    shutil.rmtree(staging_dir)

    size_mb = os.path.getsize(zip_file) / (1024 * 1024)
    print("\n" + "=" * 60)
    print("ZIP ARCHIVE CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Archive file: {zip_file}")
    print(f"Archive size: {size_mb:.2f} MB")

    return zip_file


def main():
    parser = argparse.ArgumentParser(description="Package Chemistry Multi-Agent outputs into a zip archive.")
    parser.add_argument("--dir", type=str, default=OUTPUT_DIR, help="Outputs directory.")
    parser.add_argument("--zip", type=str, default=ZIP_OUTPUT_PATH, help="Base path for zip archive (without .zip).")
    args = parser.parse_args()

    package_outputs(output_dir=args.dir, zip_base_name=args.zip)


if __name__ == "__main__":
    main()
