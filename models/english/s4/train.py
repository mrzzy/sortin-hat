#
# Sort'in Hat
# Models: English/S4
# Train Model
#

import re

from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Train English/S4")
    
    parser.add_argument(
        "data_dir", help="Directory to source model training data from."
    )
    parser.add_argument(
        "--data-regex", type=re.compile,
        default=r"Template for Data Extraction (DL Project) \d{4} Sec 4 cohort\s+Sendout.csv",
        description="Regular expression matching 'Data Extraction' CSV files.",
    )
    parser.add_argument(
        "--p6-regex", type=re.compile,
        default=r"P6 Screening Master Template (\d\d \w{3} \d{4})_Last Sendout.csv",
        description="Regular expression matching 'P6 Screening' CSV files.",
    )
    parser.parse_args()
