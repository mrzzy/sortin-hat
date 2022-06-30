#
# Sort'in Hat
# Models: English/S4
# Train Model
#

import re

from pathlib import Path
from argparse import ArgumentParser


if __name__ == "__main__":
    # parse command line arguments
    parser = ArgumentParser(description="Train English/S4")
    
    parser.add_argument(
        "DATA_DIR", type=Path,
        help="Directory to source model training data from.",
    )
    parser.add_argument(
        "--data-regex", type=re.compile,
        default=r"Template for Data Extraction \(DL Project\) \d{4} Sec4 cohort\s+Sendout.csv",
        help="Regular expression matching 'Data Extraction' CSV files.",
    )
    parser.add_argument(
        "--p6-regex", type=re.compile,
        default=r"P6 Screening Master Template \(\d\d? \w\w\w \d{4}\)_Last Sendout.csv",
        help="Regular expression matching 'P6 Screening' CSV files.",
    )
    args = parser.parse_args()
    
    if not args.DATA_DIR.is_dir():
        raise ValueError("Expected DATA_DIR to be a path to valid directory.")

    # extract absolute paths to matching data files
    paths = [p.resolve(strict=True) for p in args.DATA_DIR.rglob("*")]
    data_paths = [p for p in paths if args.data_regex.match(p.name)]
    p6_paths = [p for p in paths if args.p6_regex.match(p.name)]
