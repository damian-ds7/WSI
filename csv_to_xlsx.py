import argparse
from pathlib import Path

import pandas as pd


def convert_to_scientific_if_large(value: str, treshhold=1e6):
    value = value.replace(",", ".")

    number = float(value)

    if abs(number) >= treshhold:
        return f"{number:.2e}".replace(".", ",")

    return value.replace(".", ",")


def convert_to_single_xlsx(source_path: Path, save_path: Path):
    csv_files = list(source_path.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in the source directory.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(save_path, engine="xlsxwriter") as writer:
        for file in csv_files:
            df = pd.read_csv(file)
            sheet_name = file.stem

            for col in df.columns[1:]:
                df[col] = df[col].apply(convert_to_scientific_if_large)

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Excel file saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all CSV files in a directory to an Excel file."
    )
    parser.add_argument(
        "--source", required=True, help="Path to the directory containing CSV files."
    )
    parser.add_argument(
        "--save", required=True, help="Path to save the output Excel file."
    )

    args = parser.parse_args()

    convert_to_single_xlsx(Path(args.source), Path(args.save))
