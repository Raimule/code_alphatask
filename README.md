# code_alphatask

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns


# ==========================================
# PROJECT SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

LOCAL_DATA_PATH = DATA_DIR / "unemployment.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_unemployment.csv"
SOURCE_PATH = DATA_DIR / "dataset_source.txt"

WORLD_BANK_API = "https://api.worldbank.org/v2"
START_YEAR = 2010
END_YEAR = 2025
REQUEST_TIMEOUT = 25

INDICATORS = {
    "SL.UEM.TOTL.ZS": "Unemployment Rate (%)",
    "SL.TLF.CACT.ZS": "Labour Participation Rate (%)",
    "SL.EMP.TOTL.SP.ZS": "Employment to Population Ratio (%)",
}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 180,
    "axes.titleweight": "bold"
})


# ==========================================
# CREATE PROJECT FOLDERS
# ==========================================

def ensure_folders():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_source_note(text):
    SOURCE_PATH.write_text(text.strip() + "\n", encoding="utf-8")


# ==========================================
# DOWNLOAD WORLD BANK DATA
# ==========================================

def fetch_world_bank_indicator(indicator_code):
    url = f"{WORLD_BANK_API}/country/all/indicator/{indicator_code}"

    params = {
        "date": f"{START_YEAR}:{END_YEAR}",
        "format": "json",
        "per_page": 20000,
        "page": 1
    }

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError(
            f"No observations found for {indicator_code}"
        )

    metadata, records = payload[0], payload[1]
    all_records = list(records)

    pages = int(metadata.get("pages", 1))

    for page in range(2, pages + 1):
        params["page"] = page

        page_response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        page_response.raise_for_status()
        page_payload = page_response.json()

        if isinstance(page_payload, list) and len(page_payload) > 1:
            if page_payload[1]:
                all_records.extend(page_payload[1])

    return all_records


def download_world_bank_dataset():
    print("\nDownloading official World Bank unemployment data...")

    response = requests.get(
        f"{WORLD_BANK_API}/country",
        params={
            "format": "json",
            "per_page": 400
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise ValueError("Could not retrieve country information.")

    countries = {}

    for item in payload[1]:
        region_data = item.get("region") or {}
        region_name = str(region_data.get("value", "")).strip()
        iso3 = str(item.get("id", "")).strip()

        if iso3 and region_name and region_name.lower() != "aggregates":
            countries[iso3] = {
                "Country": item.get("name", iso3),
                "World Bank Region": region_name
            }

    merged = None

    for code, output_name in INDICATORS.items():

        try:
            records = fetch_world_bank_indicator(code)

        except Exception as error:
            if code == "SL.UEM.TOTL.ZS":
                raise

            print(f"Optional indicator unavailable: {error}")
            continue

        rows = []

        for record in records:

            iso3 = str(
                record.get("countryiso3code", "")
            ).strip()

            if iso3 not in countries:
                continue

            value = record.get("value")

            if value is None:
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            rows.append({
                "Country ISO3": iso3,
                "Country": countries[iso3]["Country"],
                "World Bank Region": countries[iso3]["World Bank Region"],
                "Year": int(record["date"]),
                output_name: numeric_value
            })

        indicator_df = pd.DataFrame(rows)

        if indicator_df.empty:
            if code == "SL.UEM.TOTL.ZS":
                raise ValueError(
                    "No usable unemployment observations found."
                )
            continue

        keys = [
            "Country ISO3",
            "Country",
            "World Bank Region",
            "Year"
        ]

        if merged is None:
            merged = indicator_df
        else:
            merged = pd.merge(
                merged,
                indicator_df,
                on=keys,
                how="outer"
            )

    if merged is None or "Unemployment Rate (%)" not in merged.columns:
        raise ValueError("No unemployment data was downloaded.")

    merged["Region"] = merged["Country"]

    merged["Date"] = pd.to_datetime(
        merged["Year"].astype(str) + "-01-01",
        errors="coerce"
    )

    merged.to_csv(LOCAL_DATA_PATH, index=False)

    write_source_note(
        "Source: World Bank Indicators API, World Development Indicators. "
        "Unemployment indicator SL.UEM.TOTL.ZS (ILO modeled estimate). "
        "Optional labour participation and employment ratio indicators "
        "were downloaded where available. World Bank aggregate rows excluded. "
        f"Requested years: {START_YEAR}-{END_YEAR}."
    )

    return merged


# ==========================================
# IDENTIFY DATASET COLUMNS
# ==========================================

def find_column(columns, candidates):

    def normalize(value):
        return "".join(
            char.lower()
            for char in str(value)
            if char.isalnum()
        )

    lookup = {
        normalize(column): column
        for column in columns
    }

    for candidate in candidates:

        key = normalize(candidate)

        if key in lookup:
            return lookup[key]

    return None


# ==========================================
# LOAD DATASET
# ==========================================

def load_input_dataset(csv_path=None):

    if csv_path is not None:

        path = csv_path.expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"CSV file not found: {path}"
            )

        raw = pd.read_csv(path)

        if path == SAMPLE_DATA_PATH.resolve():

            note = (
                "SYNTHETIC DEMONSTRATION DATA ONLY. "
                "These example records are not real unemployment statistics."
            )

        else:
            note = f"User-provided CSV: {path}"

        return raw, note

    if LOCAL_DATA_PATH.is_file():

        raw = pd.read_csv(LOCAL_DATA_PATH)

        source_note_is_current = (
            SOURCE_PATH.exists()
            and SOURCE_PATH.stat().st_mtime_ns
            >= LOCAL_DATA_PATH.stat().st_mtime_ns
        )

        if source_note_is_current:
            note = SOURCE_PATH.read_text(encoding="utf-8").strip()
        else:
            note = "User-provided CSV. Source not independently verified."

        return raw, note

    try:

        raw = download_world_bank_dataset()

        note = SOURCE_PATH.read_text(
            encoding="utf-8"
        ).strip()

        return raw, note

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError
    ) as error:

        print(f"\nWorld Bank download unavailable: {error}")

        if not SAMPLE_DATA_PATH.is_file():
            raise RuntimeError(
                "No dataset available. Add data/unemployment.csv "
                "or restore the sample dataset."
            ) from error

        raw = pd.read_csv(SAMPLE_DATA_PATH)

        note = (
            "SYNTHETIC DEMONSTRATION DATA ONLY. "
            "These records are artificial and must not be presented "
            "as actual economic findings."
        )

        write_source_note(note)

        return raw, note


# ==========================================
# DATA CLEANING
# ==========================================

def clean_dataset(raw):

    if raw.empty:
        raise ValueError("Dataset is empty.")

    data = raw.copy()

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    data = data.loc[
        :,
        ~pd.Index(data.columns).duplicated()
    ].copy()

    metadata = {
        "input_rows": len(data),
        "input_columns": len(data.columns),
        "duplicates_removed": 0,
        "invalid_core_rows_removed": 0
    }

    region_col = find_column(
        list(data.columns),
        [
            "Region",
            "State",
            "Country",
            "State/Region",
            "Region/State",
            "Area Name"
        ]
    )

    date_col = find_column(
        list(data.columns),
        [
            "Date",
            "Period",
            "Observation Date",
            "Time"
        ]
    )

    year_col = find_column(
        list(data.columns),
        ["Year", "Calendar Year"]
    )

    unemployment_col = find_column(
        list(data.columns),
        [
            "Unemployment Rate (%)",
            "Estimated Unemployment Rate (%)",
            "Unemployment Rate",
            "Unemployment_Rate",
            "Unemployment Rate Percent",
            "SL.UEM.TOTL.ZS"
        ]
    )

    employed_col = find_column(
        list(data.columns),
        [
            "Estimated Employed",
            "Employed",
            "Employment",
            "Employment Count"
        ]
    )

    participation_col = find_column(
        list(data.columns),
        [
            "Labour Participation Rate (%)",
            "Estimated Labour Participation Rate (%)",
            "Labor Participation Rate (%)",
            "Labour Force Participation Rate (%)",
            "SL.TLF.CACT.ZS"
        ]
    )

    employment_ratio_col = find_column(
        list(data.columns),
        [
            "Employment to Population Ratio (%)",
            "Employment Population Ratio (%)",
            "SL.EMP.TOTL.SP.ZS"
        ]
    )

    area_col = find_column(
        list(data.columns),
        ["Area", "Sector", "Urban/Rural"]
    )

    wb_region_col = find_column(
        list(data.columns),
        ["World Bank Region", "WB Region"]
    )

    if unemployment_col is None:
        raise ValueError(
            "Unemployment rate column not found. "
            "Expected 'Estimated Unemployment Rate (%)' "
            "or 'Unemployment Rate (%)'."
        )

    if region_col and region_col != "Region":
        data.rename(
            columns={region_col: "Region"},
            inplace=True
        )

    elif region_col is None:
        data["Region"] = "All observations"

    if date_col and date_col != "Date":
        data.rename(
            columns={date_col: "Date"},
            inplace=True
        )

    elif date_col is None and year_col:
        data["Date"] = (
            data[year_col].astype("string") + "-01-01"
        )

    elif date_col is None:
        raise ValueError(
            "Date or Year column not found."
        )

    rename_columns = {
        unemployment_col: "Unemployment Rate (%)"
    }

    if employed_col:
        rename_columns[employed_col] = "Estimated Employed"

    if participation_col:
        rename_columns[participation_col] = (
            "Labour Participation Rate (%)"
        )

    if employment_ratio_col:
        rename_columns[employment_ratio_col] = (
            "Employment to Population Ratio (%)"
        )

    if area_col:
        rename_columns[area_col] = "Area"

    if wb_region_col:
        rename_columns[wb_region_col] = "World Bank Region"

    data.rename(
        columns=rename_columns,
        inplace=True
    )

    data["Region"] = (
        data["Region"]
        .astype("string")
        .str.strip()
    )

    data["Region"] = data["Region"].replace(
        {"": pd.NA, "nan": pd.NA}
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
        dayfirst=True
    )

    numeric_columns = [
        "Unemployment Rate (%)",
        "Estimated Employed",
        "Labour Participation Rate (%)",
        "Employment to Population Ratio (%)"
    ]

    for column in numeric_columns:

        if column in data.columns:

            values = (
                data[column]
                .astype("string")
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
            )

            data[column] = pd.to_numeric(
                values,
                errors="coerce"
            )

    duplicate_count = int(data.duplicated().sum())

    data = data.drop_duplicates().copy()

    metadata["duplicates_removed"] = duplicate_count

    before_filter = len(data)

    valid_rate = data[
        "Unemployment Rate (%)"
    ].between(0, 100)

    valid_rows = (
        data["Date"].notna()
        & data["Unemployment Rate (%)"].notna()
        & valid_rate
    )

    data = data.loc[valid_rows].copy()

    data["Region"] = data["Region"].fillna("Unknown")

    metadata["invalid_core_rows_removed"] = (
        before_filter - len(data)
    )

    if data.empty:
        raise ValueError(
            "No valid records remain after cleaning. "
            "Check the date and unemployment-rate columns."
        )

    data["Year"] = data["Date"].dt.year

    if "Area" in data.columns:
        data["Area"] = (
            data["Area"].astype("string").str.strip()
        )

    metadata["missing_values_after"] = (
        data.isna().sum().to_dict()
    )

    metadata["output_rows"] = len(data)
    metadata["output_columns"] = len(data.columns)
    metadata["columns"] = list(data.columns)

    # Detect outliers but retain them.
    q1 = data["Unemployment Rate (%)"].quantile(0.25)
    q3 = data["Unemployment Rate (%)"].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = ~data[
        "Unemployment Rate (%)"
    ].between(lower, upper)

    metadata["outlier_count"] = int(
        outlier_mask.sum()
    )

    metadata["outlier_lower_bound"] = float(lower)
    metadata["outlier_upper_bound"] = float(upper)

    metadata["outlier_rows"] = data.loc[
        outlier_mask
    ].copy()

    return (
        data.sort_values("Date").reset_index(drop=True),
        metadata
    )


# ==========================================
# COVID-19 ANALYSIS
# ==========================================

def period_name(year):

    if 2015 <= year <= 2019:
        return "Before COVID-19 (2015–2019)"

    if year in (2020, 2021):
        return "During COVID-19 (2020–2021)"

    if 2022 <= year <= 2025:
        return "After COVID-19 (2022–2025)"

    return "Outside comparison window"


def covid_summary(data):

    covid = data.copy()

    covid["COVID Period"] = (
        covid["Year"].map(period_name)
    )

    covid = covid[
        covid["COVID Period"] != "Outside comparison window"
    ]

    if covid.empty:
        return pd.DataFrame(
            columns=[
                "COVID Period",
                "Average Unemployment Rate (%)",
                "Records"
            ]
        )

    order = [
        "Before COVID-19 (2015–2019)",
        "During COVID-19 (2020–2021)",
        "After COVID-19 (2022–2025)"
    ]

    result = covid.groupby("COVID Period").agg(
        **{
            "Average Unemployment Rate (%)": (
                "Unemployment Rate (%)", "mean"
            ),
            "Records": (
                "Unemployment Rate (%)", "size"
            )
        }
    )

    return (
        result.reindex(order)
        .dropna(subset=["Average Unemployment Rate (%)"])
        .reset_index()
    )


def covid_region_summary(data):

    covid = data.copy()

    covid["COVID Period"] = (
        covid["Year"].map(period_name)
    )

    covid = covid[
        covid["COVID Period"] != "Outside comparison window"
    ]

    if covid.empty:
        return pd.DataFrame()

    return covid.groupby(
        ["Region", "COVID Period"],
        as_index=False
    ).agg(
        **{
            "Average Unemployment Rate (%)": (
                "Unemployment Rate (%)", "mean"
            ),
            "Records": (
                "Unemployment Rate (%)", "size"
            )
        }
    )


# ==========================================
# DATA VISUALIZATION
# ==========================================

def create_visualizations(data):

    created = []

    # 1. Unemployment over time
    trend = data.groupby(
        "Date", as_index=False
    )["Unemployment Rate (%)"].mean()

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.lineplot(
        data=trend,
        x="Date",
        y="Unemployment Rate (%)",
        marker="o",
        ax=ax
    )

    ax.set_title("Average Unemployment Rate Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment rate (%)")

    fig.autofmt_xdate()
    fig.tight_layout()

    path = OUTPUT_DIR / "unemployment_over_time.png"

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    created.append(path.name)

    # Average unemployment by region
    averages = data.groupby(
        "Region"
    )["Unemployment Rate (%)"].mean().sort_values(
        ascending=False
    )

    top_averages = averages.head(15).sort_values()

    fig, ax = plt.subplots(
        figsize=(10, max(5, len(top_averages) * 0.36))
    )

    sns.barplot(
        x=top_averages.values,
        y=top_averages.index,
        color="#4776A8",
        ax=ax
    )

    ax.set_title("Average Unemployment Rate by Region")
    ax.set_xlabel("Average unemployment rate (%)")
    ax.set_ylabel("Region")

    fig.tight_layout()

    path = OUTPUT_DIR / "average_unemployment.png"

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    created.append(path.name)

    # Highest and lowest regions
    if len(averages) > 10:
        high_low = pd.concat(
            [averages.head(5), averages.tail(5)]
        ).sort_values()
    else:
        high_low = averages.sort_values()

    fig, ax = plt.subplots(
        figsize=(10, max(5, len(high_low) * 0.42))
    )

    sns.barplot(
        x=high_low.values,
        y=high_low.index,
        color="#9876A8",
        ax=ax
    )

    ax.set_title("Highest and Lowest Unemployment Regions")
    ax.set_xlabel("Average unemployment rate (%)")
    ax.set_ylabel("Region")

    fig.tight_layout()

    path = OUTPUT_DIR / "highest_lowest_regions.png"

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    created.append(path.name)

    # COVID comparison
    covid = covid_summary(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    if not covid.empty:

        sns.barplot(
            data=covid,
            x="COVID Period",
            y="Average Unemployment Rate (%)",
            color="#D08B42",
            ax=ax
        )

        ax.tick_params(axis="x", rotation=15)

    else:

        ax.text(
            0.5, 0.5,
            "No data available for COVID comparison",
            ha="center",
            va="center"
        )

        ax.set_xticks([])

    ax.set_title(
        "Unemployment Before, During and After COVID-19"
    )

    ax.set_xlabel("Comparison period")
    ax.set_ylabel("Average unemployment rate (%)")

    fig.tight_layout()

    path = OUTPUT_DIR / "covid_analysis.png"

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    created.append(path.name)

    # Regional or area comparison
    compare_col = None

    if (
        "Area" in data.columns
        and data["Area"].nunique(dropna=True) > 1
    ):
        compare_col = "Area"

    elif (
        "World Bank Region" in data.columns
        and data["World Bank Region"].nunique(dropna=True) > 1
    ):
        compare_col = "World Bank Region"

    fig, ax = plt.subplots(figsize=(10, 6))

    if compare_col:

        compare = data.groupby(
            compare_col
        )["Unemployment Rate (%)"].mean().sort_values(
            ascending=False
        )

        sns.barplot(
            x=compare.values,
            y=compare.index.astype(str),
            color="#4B9B83",
            ax=ax
        )

        ax.set_title(
            f"Unemployment Comparison by {compare_col}"
        )

        ax.set_xlabel("Average unemployment rate (%)")
        ax.set_ylabel(compare_col)

    else:

        yearly = data.groupby(
            "Year"
        )["Unemployment Rate (%)"].mean()

        sns.barplot(
            x=yearly.index.astype(str),
            y=yearly.values,
            color="#4B9B83",
            ax=ax
        )

        ax.set_title("Yearly Unemployment Comparison")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average unemployment rate (%)")

    fig.tight_layout()

    path = OUTPUT_DIR / "regional_comparison.png"

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    created.append(path.name)

    # Unemployment distribution
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.histplot(
        data=data,
        x="Unemployment Rate (%)",
        bins=20,
        kde=True,
        color="#527DA8",
        ax=ax
    )

    ax.set_title("Distribution of Unemployment Rates")
    ax.set_xlabel("Unemployment rate (%)")
    ax.set_ylabel("Number of observations")

    fig.tight_layout()

    path = OUTPUT_DIR / "unemployment_distribution.png"

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    created.append(path.name)

    # Additional analysis
    fig, ax = plt.subplots(figsize=(9, 6))

    if (
        "Labour Participation Rate (%)" in data.columns
        and data["Labour Participation Rate (%)"].notna().sum() >= 2
    ):

        sns.scatterplot(
            data=data,
            x="Labour Participation Rate (%)",
            y="Unemployment Rate (%)",
            alpha=0.65,
            ax=ax
        )

        ax.set_title(
            "Unemployment vs Labour Participation Rate"
        )

        ax.set_xlabel("Labour participation rate (%)")
        ax.set_ylabel("Unemployment rate (%)")

    else:

        counts = data.groupby("Year").size()

        sns.barplot(
            x=counts.index.astype(str),
            y=counts.values,
            color="#9876A8",
            ax=ax
        )

        ax.set_title("Number of Records by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of records")

    fig.tight_layout()

    path = OUTPUT_DIR / "additional_analysis.png"

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    created.append(path.name)

    return created


# ==========================================
# GENERATE ANALYSIS REPORT
# ==========================================

def generate_report(data, metadata, source_note):

    rates = data["Unemployment Rate (%)"]

    region_means = data.groupby(
        "Region"
    )["Unemployment Rate (%)"].mean().sort_values(
        ascending=False
    )

    timeline = data.groupby(
        "Date"
    )["Unemployment Rate (%)"].mean().sort_index()

    covid = covid_summary(data)

    lines = [
        "UNEMPLOYMENT ANALYSIS WITH PYTHON",
        "=" * 64,
        "",
        f"Dataset source: {source_note}",
        f"Records after cleaning: {len(data)}",
        f"Regions/groups: {data['Region'].nunique()}",
        f"Date coverage: {data['Date'].min().date()} to {data['Date'].max().date()}",
        "",
        "DATA QUALITY",
        "-" * 64,
        f"Input rows: {metadata['input_rows']}",
        f"Input columns: {metadata['input_columns']}",
        f"Duplicates removed: {metadata['duplicates_removed']}",
        f"Invalid rows removed: {metadata['invalid_core_rows_removed']}",
        f"Potential outliers retained: {metadata['outlier_count']}",
        "",
        "DESCRIPTIVE STATISTICS",
        "-" * 64,
        f"Average unemployment rate: {rates.mean():.2f}%",
        f"Median unemployment rate: {rates.median():.2f}%",
        f"Maximum unemployment rate: {rates.max():.2f}%",
        f"Minimum unemployment rate: {rates.min():.2f}%",
        f"Total valid records: {len(data)}",
        f"Highest average region: {region_means.index[0]} ({region_means.iloc[0]:.2f}%)",
        f"Lowest average region: {region_means.index[-1]} ({region_means.iloc[-1]:.2f}%)"
    ]

    if (
        "Estimated Employed" in data.columns
        and data["Estimated Employed"].notna().any()
    ):

        lines.append(
            f"Average estimated employed: "
            f"{data['Estimated Employed'].mean():,.2f}"
        )

    else:

        lines.append(
            "Estimated employed: Not available."
        )

    if (
        "Labour Participation Rate (%)" in data.columns
        and data["Labour Participation Rate (%)"].notna().any()
    ):

        lines.append(
            f"Average labour participation rate: "
            f"{data['Labour Participation Rate (%)'].mean():.2f}%"
        )

    else:

        lines.append(
            "Labour participation rate: Not available."
        )

    if (
        "Employment to Population Ratio (%)" in data.columns
        and data["Employment to Population Ratio (%)"].notna().any()
    ):

        lines.append(
            f"Average employment-to-population ratio: "
            f"{data['Employment to Population Ratio (%)'].mean():.2f}%"
        )

    lines.extend([
        "",
        "TREND OBSERVATIONS",
        "-" * 64
    ])

    if len(timeline) >= 2:

        change = timeline.iloc[-1] - timeline.iloc[0]

        direction = (
            "increased" if change > 0
            else "decreased" if change < 0
            else "remained unchanged"
        )

        lines.append(
            f"The average rate {direction} by "
            f"{abs(change):.2f} percentage points between "
            f"{timeline.index[0].date()} and "
            f"{timeline.index[-1].date()}."
        )

        changes = timeline.diff().dropna()

        if not changes.empty:

            biggest_up = changes.idxmax()
            biggest_down = changes.idxmin()

            lines.append(
                f"Largest increase: {changes.max():.2f} "
                f"percentage points, ending {biggest_up.date()}."
            )

            lines.append(
                f"Largest decrease: {changes.min():.2f} "
                f"percentage points, ending {biggest_down.date()}."
            )

    else:

        lines.append(
            "Not enough distinct dates to calculate a trend."
        )

    lines.extend([
        "",
        "COVID-19 PERIOD COMPARISON",
        "-" * 64
    ])

    if covid.empty:

        lines.append(
            "No records available in the defined COVID comparison periods."
        )

    else:

        for _, row in covid.iterrows():

            lines.append(
                f"{row['COVID Period']}: "
                f"{row['Average Unemployment Rate (%)']:.2f}% "
                f"({int(row['Records'])} records)."
            )

        covid_means = covid.set_index(
            "COVID Period"
        )["Average Unemployment Rate (%)"]

        before = "Before COVID-19 (2015–2019)"
        during = "During COVID-19 (2020–2021)"
        after = "After COVID-19 (2022–2025)"

        if before in covid_means.index and during in covid_means.index:

            delta = covid_means[during] - covid_means[before]

            word = (
                "higher" if delta > 0
                else "lower" if delta < 0
                else "the same as"
            )

            lines.append(
                f"During-COVID average was {abs(delta):.2f} "
                f"percentage points {word} the pre-COVID average."
            )

        if during in covid_means.index and after in covid_means.index:

            delta = covid_means[after] - covid_means[during]

            word = (
                "higher" if delta > 0
                else "lower" if delta < 0
                else "the same as"
            )

            lines.append(
                f"Post-COVID average was {abs(delta):.2f} "
                f"percentage points {word} the during-COVID average."
            )

    lines.extend([
        "",
        "INTERPRETATION AND LIMITATIONS",
        "-" * 64,
        "These descriptive comparisons do not prove that COVID-19 caused changes.",
        "Other economic, policy and demographic factors may contribute.",
        "Period averages are simple averages unless suitable weights are supplied.",
        "Missing information is not invented."
    ])

    if "SYNTHETIC DEMONSTRATION DATA ONLY" in source_note:

        lines.extend([
            "",
            "IMPORTANT WARNING",
            "This report uses synthetic demonstration data.",
            "Numerical results are artificial and must not be "
            "presented as real-world findings."
        ])

    else:

        lines.extend([
            "",
            "POSSIBLE POLICY-RELEVANT OBSERVATIONS",
            "-" * 64,
            "Investigate regions with persistently higher unemployment.",
            "Compare unemployment with labour participation.",
            "Review trends across multiple periods before designing interventions."
        ])

    report_text = "\n".join(lines) + "\n"

    report_path = OUTPUT_DIR / "analysis_report.txt"

    report_path.write_text(
        report_text,
        encoding="utf-8"
    )

    return report_text


# ==========================================
# COMPLETE DATA ANALYSIS
# ==========================================

def run_analysis(csv_path=None):

    ensure_folders()

    raw, source_note = load_input_dataset(csv_path)

    print("\nFIRST FIVE RECORDS")
    print(raw.head().to_string(index=False))

    print("\nDATASET SHAPE")
    print(raw.shape)

    print("\nCOLUMN DATA TYPES")
    print(raw.dtypes)

    print("\nMISSING VALUES")
    print(raw.isna().sum())

    clean, metadata = clean_dataset(raw)

    clean.to_csv(
        OUTPUT_DIR / "cleaned_data.csv",
        index=False
    )

    regional_covid = covid_region_summary(clean)

    if not regional_covid.empty:

        regional_covid.to_csv(
            OUTPUT_DIR / "covid_region_comparison.csv",
            index=False
        )

    if not metadata["outlier_rows"].empty:

        metadata["outlier_rows"].to_csv(
            OUTPUT_DIR / "potential_outliers.csv",
            index=False
        )

    print("\nDATA CLEANING SUMMARY")

    print(
        "Duplicates removed:",
        metadata["duplicates_removed"]
    )

    print(
        "Invalid rows removed:",
        metadata["invalid_core_rows_removed"]
    )

    print(
        "Potential outliers retained:",
        metadata["outlier_count"]
    )

    # Statistics
    rates = clean["Unemployment Rate (%)"]

    print("\nUNEMPLOYMENT STATISTICS")

    print(f"Average: {rates.mean():.2f}%")
    print(f"Maximum: {rates.max():.2f}%")
    print(f"Minimum: {rates.min():.2f}%")
    print(f"Median: {rates.median():.2f}%")

    print(f"Total records: {len(clean)}")

    averages = clean.groupby(
        "Region"
    )["Unemployment Rate (%)"].mean()

    print(
        "Highest unemployment region:",
        averages.idxmax()
    )

    print(
        "Lowest unemployment region:",
        averages.idxmin()
    )

    if "Estimated Employed" in clean.columns:

        print(
            "Average estimated employed:",
            clean["Estimated Employed"].mean()
        )

    if "Labour Participation Rate (%)" in clean.columns:

        print(
            "Average labour participation:",
            clean["Labour Participation Rate (%)"].mean()
        )

    # Generate visualizations
    charts = create_visualizations(clean)

    # Generate report
    report = generate_report(
        clean,
        metadata,
        source_note
    )

    print("\nCOVID-19 ANALYSIS")

    covid = covid_summary(clean)

    if covid.empty:
        print("No COVID comparison data available.")
    else:
        print(covid.to_string(index=False))

    print("\nVISUALIZATIONS CREATED")

    for chart in charts:
        print(f"outputs/{chart}")

    print("\nREPORT CREATED")
    print("outputs/analysis_report.txt")

    print("\nCLEANED DATA CREATED")
    print("outputs/cleaned_data.csv")

    print("\nANALYSIS COMPLETED SUCCESSFULLY!")

    return {
        "data": clean,
        "charts": charts,
        "report": report,
        "metadata": metadata,
        "source": source_note
    }


# ==========================================
# COMMAND LINE ARGUMENTS
# ==========================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Unemployment Analysis with Python"
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to an unemployment CSV dataset."
    )

    return parser.parse_args()


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    try:

        args = parse_args()

        print("=" * 65)
        print("CODEALPHA TASK 2")
        print("UNEMPLOYMENT ANALYSIS WITH PYTHON")
        print("=" * 65)

        print("Python version:", sys.version.split()[0])

        result = run_analysis(args.csv)

        print("\nProject execution completed.")

        if "SYNTHETIC DEMONSTRATION DATA ONLY" in result["source"]:

            print(
                "\nWARNING: Sample demonstration data was used. "
                "Add a real dataset before submitting actual findings."
            )

        return 0

    except FileNotFoundError as error:

        print("\nFile error:", error)
        return 1

    except (
        ValueError,
        RuntimeError,
        pd.errors.ParserError
    ) as error:

        print("\nData error:", error)
        return 1

    except KeyboardInterrupt:

        print("\nProgram stopped by user.")
        return 1

    except Exception as error:

        logging.exception("Unexpected project error")

        print("\nUnexpected error:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
