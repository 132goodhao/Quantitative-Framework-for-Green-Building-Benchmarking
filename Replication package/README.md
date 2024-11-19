# Overview

This replication package supports the reproduction of the results presented in the study titled **"Unveiling the Impact of Green Building Benchmarking Policy: Evidence from Singapore"**. It provides access to the necessary data, scripts, and a step-by-step guide to process the data, perform the analyses, and replicate the results. Welcome to visit our GitHub repository:

[GitHub - Quantitative Framework](https://github.com/132goodhao/Quantitative-Framework-for-Green-Building-Benchmarking/tree/main)

# 1. Data

## (1) How to get the data?

### Online Sources

The primary data is sourced from the Singapore Building and Construction Authority (BCA). These data include benchmarking reports and energy performance metrics. The current data are available on the updated BCA website:

> [BCA Building Energy Benchmarking and Disclosure](https://www1.bca.gov.sg/buildsg/sustainability/regulatory-requirements-for-existing-buildings/bca-building-energy-benchmarking-and-disclosure)

### Included in this Package

The data files used in the study are also provided in this replication package under the `01_Original` directory. They can be directly accessed from the GitHub repository:

> [GitHub - Data Folder](https://github.com/132goodhao/Quantitative-Framework-for-Green-Building-Benchmarking/tree/main/data)

## (2) How to process the data?

### Data Processing Steps

Data processing involves cleaning, formatting, and encoding steps. Below is an overview of the scripts used for each stage and their corresponding output directories:

| **Script**                   | **Purpose**                                                                                             | **Output Directory**     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------ |
| `Unified_and_Cleaned.py`     | Removes samples with missing `Address` and merges all EUI data. Calculates GM Version distribution.     | `06_Unified_and_Cleaned` |
| `Merge_and_unify.py`         | Standardizes categories for `Type` and `Function`. Unifies variables and converts units to percentages. | `05_Merged_and_unified`  |
| `Merge_all_files.py`         | Attempts to merge all datasets into a single file.                                                      | `04_Reformed_and_Merged` |
| N/A                          | Further refines formats and removes redundant variables.                                                | `03_Reformed_Reformed`   |
| `Rename_the_title_failed.py` | Standardizes variable names, categories, and counts.                                                    | `02_Reformed_Original`   |
| N/A                          | Original data without processing.                                                                       | `01_Original`            |

# 2. Model

## (1) Correlation Analysis

**This analysis calculates:**

- **Pearson Correlation Coefficients:** Measures linear relationships between variables.
- **Time-Fixed Effects:** Evaluates correlations with adjustments for time-invariant factors.

**Script:**

The analysis is implemented in:  
`00_Drafts/07_Correlation Analysis.py`

**Outputs:**

- Heatmaps and scatterplots showing relationships between variables.
- Summary tables of correlation coefficients.

## (2) PS-Matching

The Propensity Score Matching (PS-Matching) methodology involves:

- **Pre-Matching Analysis:** Prepares the data and checks for balance in covariates.
- **Matching and Post-Matching Analysis:** Matches samples and assesses the treatment effect.

**Scripts:**

- **Pre-Matching:**  
  `00_Drafts/08_Preparing for models.py`
- **Matching and Post-Matching:**  
  `00_Drafts/09_Treatment_Effect_of_PS.py`
- **Environmental and Economic Assessment:**  
  `00_Drafts/10_Environmental_Economic_Assessment.py`

**Unique Features:**

Unlike many existing PS-matching implementations, our scripts include:

- Pre-matching balance checks.
- Post-matching validation tests.
- Comprehensive assessment of statistical assumptions.

**Outputs:**

- Treatment effect estimates (ATE, ATET, ATENT).
- Visualizations of covariate balance and matching results.
- Environmental and economic impact analysis.
