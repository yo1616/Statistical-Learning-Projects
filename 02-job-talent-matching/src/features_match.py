import re
import pandas as pd


def _extract_first_token(text: str) -> str:
    return str(text).split(" ")[0]


def _extract_numbers(text: str):
    nums = re.findall(r"\d+(?:\.\d+)?", str(text))
    return [float(n) for n in nums] if nums else []


def _parse_range(text: str):
    nums = _extract_numbers(text)
    if not nums:
        return None, None, None
    if len(nums) == 1:
        return nums[0], nums[0], nums[0]
    lo, hi = min(nums), max(nums)
    return lo, hi, (lo + hi) / 2.0


def _overlap_count(a: str, b: str) -> int:
    tokens_a = set(re.split(r"[ ,;/]", str(a)))
    tokens_b = set(re.split(r"[ ,;/]", str(b)))
    tokens_a.discard("")
    tokens_b.discard("")
    return len(tokens_a & tokens_b)


def add_matching_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    增加 matching 特徵與部分數值特徵。
    """
    df = df.copy()

    # 地點匹配
    if {"job_workcity", "talent_preference_city"}.issubset(df.columns):
        df["match_city"] = df.apply(
            lambda row: int(_extract_first_token(row["job_workcity"]) in str(row["talent_preference_city"])), axis=1
        )

    # 班別匹配
    if {"job_worktime", "talent_preference_worktime"}.issubset(df.columns):
        df["match_worktime"] = (df["job_worktime"] == df["talent_preference_worktime"]).astype(int)

    # 休假匹配
    if {"job_vacation", "talent_preference_vacation"}.issubset(df.columns):
        df["match_vacation"] = (df["job_vacation"] == df["talent_preference_vacation"]).astype(int)

    # 職務/偏好 duty 交集
    if {"job_duty", "talent_preference_duty0"}.issubset(df.columns):
        df["match_duty_overlap"] = df.apply(
            lambda row: _overlap_count(row["job_duty"], row["talent_preference_duty0"]), axis=1
        )

    # 技能交集
    if {"job_matter", "talent_structured_skill"}.issubset(df.columns):
        df["match_skill_overlap"] = df.apply(
            lambda row: _overlap_count(row["job_matter"], row["talent_structured_skill"]), axis=1
        )

    # 薪資數值化
    if "job_salary" in df.columns:
        lo, hi, mean = zip(*df["job_salary"].map(_parse_range))
        df["salary_min"] = lo
        df["salary_max"] = hi
        df["salary_mean"] = mean

    # 年齡數值化
    if "job_age" in df.columns:
        lo, hi, mean = zip(*df["job_age"].map(_parse_range))
        df["age_min"] = lo
        df["age_max"] = hi

    # 公司資本與員工數
    if "org_capital" in df.columns:
        df["org_capital_num"] = df["org_capital"].map(lambda x: _extract_numbers(x)[0] if _extract_numbers(x) else None)
    if "org_staff" in df.columns:
        df["org_staff_num"] = df["org_staff"].map(lambda x: _extract_numbers(x)[0] if _extract_numbers(x) else None)

    # 文字長度特徵
    if "job_duty" in df.columns:
        df["len_job_duty"] = df["job_duty"].fillna("").str.len()
    if "talent_experience" in df.columns:
        df["len_talent_experience"] = df["talent_experience"].fillna("").str.len()
    if "talent_freeform_skill" in df.columns:
        df["len_talent_freeform"] = df["talent_freeform_skill"].fillna("").str.len()

    return df
