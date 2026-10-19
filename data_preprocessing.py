"""
Data Preprocessing Pipeline for JEE Main Chemistry Questions
Converts and cleans raw dataset into high-quality text-only questions.
"""

import os
import argparse
import numpy as np
import pandas as pd
from config import DATASET_NAME, DATASET_CONFIG, CLEANED_DATA_PATH


def is_text_only(value):
    """
    Check if a question image field represents text-only (empty/null/empty list).
    """
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, (list, np.ndarray)):
        return len(value) == 0
    if isinstance(value, str):
        val_clean = value.strip()
        return val_clean in ["", "[]", "None", "null", "NULL", "nan", "NaN"]
    return False


class ChemistryDataPreprocessor:
    """Preprocesses raw chemistry question dataset into a clean text-only format."""

    def __init__(self, dataset_name=DATASET_NAME, data_dir=DATASET_CONFIG):
        self.dataset_name = dataset_name
        self.data_dir = data_dir

    def load_dataset(self):
        """Loads dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required. Install via: pip install datasets"
            )

        print(f"Loading dataset '{self.dataset_name}' (dir='{self.data_dir}')...")
        ds = load_dataset(self.dataset_name, data_dir=self.data_dir)
        df = ds["train"].to_pandas()
        print(f"Original shape: {df.shape}")
        return df

    def run(self, output_file=CLEANED_DATA_PATH):
        """Executes the full cleaning and deduplication pipeline."""
        df = self.load_dataset()

        # 1. Discover original question ID column
        possible_id_columns = [
            "id", "question_id", "_id", "uuid", "ID", "questionId"
        ]
        id_column = None
        for col in possible_id_columns:
            if col in df.columns:
                id_column = col
                break

        if id_column is None:
            raise ValueError(f"No original question ID column found in {df.columns.tolist()}")

        print(f"Identified ID column: '{id_column}'")

        # 2. Check required columns
        required_columns = [
            id_column, "question", "solution", "subject", "topic", "difficulty", "question_images"
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # 3. Keep only text-only questions
        before = len(df)
        df = df[df["question_images"].apply(is_text_only)].copy()
        print(f"Text-only filtering: kept {len(df)} / {before} (removed {before - len(df)})")

        # 4. Remove missing / empty questions
        before = len(df)
        df = df.dropna(subset=["question"]).copy()
        df["question"] = df["question"].astype(str).str.strip()
        df = df[df["question"] != ""].copy()
        print(f"Empty question filtering: kept {len(df)} / {before} (removed {before - len(df)})")

        # 5. Deduplicate question text
        before = len(df)
        df = df.drop_duplicates(subset=["question"], keep="first").copy()
        print(f"Duplicate question text filtering: kept {len(df)} / {before} (removed {before - len(df)})")

        # 6. Deduplicate question IDs
        before = len(df)
        df = df.drop_duplicates(subset=[id_column], keep="first").copy()
        print(f"Duplicate ID filtering: kept {len(df)} / {before} (removed {before - len(df)})")

        # 7. Construct standardized final dataframe
        final_df = pd.DataFrame({
            "original_question_id": df[id_column].astype(str),
            "question": df["question"],
            "answer": df["solution"],
            "topic": df["subject"],
            "subtopic": df["topic"],
            "difficulty": df["difficulty"]
        })

        # 8. Clean columns
        for col in final_df.columns:
            final_df[col] = final_df[col].fillna("").astype(str).str.strip()

        # 9. Filter empty IDs & reset index
        final_df = final_df[final_df["original_question_id"] != ""].copy()
        final_df = final_df.reset_index(drop=True)

        # 10. Save preprocessed dataset
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        final_df.to_csv(output_file, index=False)

        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Output saved to: {output_file}")
        print(f"Final shape    : {final_df.shape}")
        print(f"Total questions: {len(final_df)}")
        print(f"Unique IDs     : {final_df['original_question_id'].nunique()}")

        return final_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess JEE Main Chemistry dataset.")
    parser.add_argument(
        "--output", type=str, default=CLEANED_DATA_PATH,
        help="Path where cleaned CSV should be saved."
    )
    args = parser.parse_args()

    preprocessor = ChemistryDataPreprocessor()
    preprocessor.run(output_file=args.output)


if __name__ == "__main__":
    main()
