#
# Sort'in Hat
# Models
# Feature Extraction
#

from math import isnan
import numpy as np
from typing import Union, cast

# feature extraction functions sourced from '01 English.ipynb'
# get psle results band
def encode_psle(score):
    if score == "A*":
        return 1
    elif score == "A":
        return 2
    elif score == "B":
        return 3
    elif score == "C":
        return 4
    elif score == "D":
        return 5
    elif score == "E":
        return 6
    elif score == "F":
        return 7
    else:
        return float("NaN")


# get housing tier
def get_housing(house):
    if house == "Detached House":
        return 1
    elif house == "Semi-Detached House":
        return 2
    elif house == "Terrace":
        return 3
    elif house == "Private Flat/Apartment":
        return 4
    elif house == "Govt/Quasi-Govt Executive Flat":
        return 5
    elif house == "HDB/SAF/PSA/PUB 5_Room Flat":
        return 6
    elif house == "HDB/SAF/PSA/PUB 4_Room Flat":
        return 7
    elif house == "HDB/SAF/PSA/PUB 3_Room Flat":
        return 8
    elif house == "other":
        return 9
    else:
        return float("NaN")


def get_course_tier(course):
    if course == "Express":
        return 1
    elif course == "Normal Academic":
        return 2
    elif course == "Normal Technical":
        return 3
    else:
        return float("NaN")


def get_gender(gender):
    if gender == "Male":
        return 1
    elif gender == "Female":
        return 0
    else:
        return float("NaN")


def get_carding(i):
    if i == True:
        return 1
    elif i == False:
        return 0
    else:
        return float("NaN")


def get_grade(score):
    """Obtain a academic score grade for the given score 0-100."""
    # TODO(mrzzy): this banding is only correct for express students
    if score >= 75:
        return 1
    elif score >= 70 and score < 75:
        return 2
    elif score >= 65 and score < 70:
        return 3
    elif score >= 60 and score < 65:
        return 4
    elif score >= 55 and score < 60:
        return 5
    elif score >= 50 and score < 55:
        return 6
    elif score >= 45 and score < 50:
        return 7
    elif score >= 40 and score < 45:
        return 8
    elif score < 40:
        return 9
    else:
        return float("NaN")


def encode_sports_level(level: Union[str, float]) -> int:
    """Encode the given Sport Level as an integer."""
    # rank missing values in the last ranking (10).
    if isinstance(level, float) and isnan(level):
        return 10

    # strip leading 'L' from level as some level's are missing leading 'L'
    level = cast(str, level).replace("L", "")

    if level == "1*":
        return 1
    elif level == "1A":
        return 2
    elif level == "1":
        return 3
    elif level == "2*":
        return 4
    elif level == "2A":
        return 5
    elif level == "2":
        return 6
    elif level == "3*":
        return 7
    elif level == "3A":
        return 8
    elif level == "3":
        return 9
    else:
        raise ValueError(f"Unsupported sports level: {level}")