# Quantitative-Framework-for-Green-Building-Benchmarking

Welcome to the repository for **"Unveiling the Impact of Green Building Benchmarking Policy: Evidence from Singapore"**. This repository includes datasets, scripts, and documentation to help you replicate the findings in the paper and extend the analysis to similar datasets or policy evaluations. Below, we provide an overview of the data, code structure, and guidance on how to use the framework effectively.

---

## Data Description

### Sources

The datasets provided here were collected from publicly available sources, primarily from the **Singapore Building and Construction Authority (BCA)**. The key source is the BCA's [Building Energy Benchmarking and Disclosure](https://www1.bca.gov.sg/buildsg/sustainability/regulatory-requirements-for-existing-buildings/bca-building-energy-benchmarking-and-disclosure).

For convenience, we have also uploaded the data used in this study to the [`data`](https://github.com/132goodhao/Quantitative-Framework-for-Green-Building-Benchmarking/tree/main/data) folder of this repository.

### Data Processing

The raw datasets underwent several processing steps to ensure consistency and reliability:

- Cleaning: Removal of records with missing or erroneous values (e.g., empty addresses or invalid metrics).
- Aggregation: Combining data from multiple sources into a unified structure.
- Transformation: Standardizing formats, applying one-hot encoding to categorical variables, and normalizing numerical features.

The data processing scripts and their corresponding output folders are as follows:

| **Script**                   | **Purpose**                                                                                                                                 | **Output Folder**        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `Unified_and_Cleaned.py`     | - Remove records with missing addresses.<br>- Merge Energy Usage Intensities (EUIs).<br>- Analyze GM version distribution.                  | `06_Unified_and_Cleaned` |
| `Merge_and_Unify.py`         | - Harmonize types (e.g., healthcare and educational categories).<br>- Adjust variables for consistency.<br>- Convert units and percentages. | `05_Merged_and_Unified`  |
| `Merge_all_files.py`         | Attempt to merge all available datasets into one unified structure.                                                                         | `04_Reformed_and_Merged` |
| `Rename_the_title_failed.py` | Standardize variable names, categories, and counts.                                                                                         | `02_Reformed_Original`   |
| N/A                          | Process raw data into an intermediate structure with consistent formatting.                                                                 | `03_Reformed_Reformed`   |
| N/A                          | Raw data as received from the source.                                                                                                       | `01_Original`            |

**Important Note:** While the data has been cleaned and prepared for analysis, please verify its suitability for your specific research needs. If you make modifications or republish the data, kindly notify us.

---

## Overview of the framework

This framework provides tools to evaluate the correlation and causality between policies and influencing factors. It includes two primary models:

1. **Fixed Effects Model (based on Elastic Net Regression):** Analyzes the relationship between policy implementation and various socioeconomic or energy-related factors.
2. **Causal Model (k-Nearest Propensity Score Matching):** Quantifies the causal impact of policies on outcomes, with robust balance checks and validation tests.

Unlike many existing PS-matching implementations, this framework emphasizes statistical rigor by incorporating:

- **Pre-matching balance checks** to ensure the validity of matches.
- **Post-matching validation tests** to confirm the robustness of causal inferences.
- **Comprehensive diagnostics** to minimize misuse of econometric techniques.

### Core code

The core codes is organized into three parts:

1. **Correlation Analysis**
   
   - Focus: Pearson correlation coefficients and time-fixed effects.
   - Script: `Correlation Analysis.py`
   - Output: Identifies significant correlations and the fixed effects of time on the dependent variables.

2. **Causality Analysis**
   
   - Focus: Pre-matching, matching, and post-matching stages of propensity score analysis.
   - Scripts:
     - `Pre-models.py` for pre-matching balance checks.
     - `Treatment Effect.py` for matching and post-matching validations.
   - Output: Provides causal estimates and balance diagnostics for policy effects.

3. **Environmental and Economic Benefit Assessment**
   
   - Focus: Estimating environmental and economic benefits from policy-induced changes.
   - Script: `Environmental Economic Assessment.py`
   - Output: Calculates energy savings, carbon emission reductions, and financial savings resulting from policy interventions.

**Flexibility:** This framework can be adapted to datasets from other regions or contexts, provided the data structure aligns with the input requirements of the scripts.

![image]()

---

## How to Use This Repository

### 1. Clone the repository

### 2. Data Preparation

- Raw data is available in the `data` folder.
- Use the scripts in the `data_processing` folder to clean, aggregate, and transform the data as required. Follow the table in the "Data Processing" section for guidance.

### 3. Analysis and Modeling

- Perform correlation analysis using the `Correlation Analysis.py` script.
- Estimate causal effects using `Pre-models.py` and `Treatment Effect.py`.
- Evaluate environmental and economic impacts with `Environmental Economic Assessment.py`.

### 4. Outputs

- Results are saved in the respective output folders mentioned in the scripts.
- Check the `results` folder for pre-generated outputs from us.

---

## Additional Notes

1. **Code Compatibility:**  
   The code is independent of any external software environments and is built entirely from the ground up. However, it assumes the input data adheres to the prescribed format.

2. **Directory Structure:**  
   Some relative paths in the scripts may differ based on your folder setup. If needed, contact us for further assistance.

3. **Ethical Use:**  
   When using or modifying this repository, please acknowledge our authors. For academic publications, cite the paper associated with this repository.
