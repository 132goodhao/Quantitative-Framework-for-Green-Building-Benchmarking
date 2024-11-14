# Quantitative-Framework-for-Green-Building-Benchmarking

## Data Description

This dataset was prepared by the author for the paper titled “**Unveiling the Impact of Green Building Benchmarking Policy: Evidence from Singapore**”. All data is sourced from public channels, and the sources have been clearly cited in the paper. Beyond the raw data, the remaining data has been cleaned and aggregated by the author to remove a significant amount of missing and erroneous values present in the original dataset. Please exercise caution when using this data to ensure it fits your research purposes. Additionally, if you plan to modify or publish any part of this dataset, please notify the author.

## Quantitative Framework Code

### Code Description

This framework is mainly used to explore the correlation and causality between policies and many influencing factors. Among them, two models are the most important: the fixed effect model (based on elastic network regression) and the causal model (k-nearest PS-matching). The paper provides a detailed explanation of the relevant mathematical principles. Furthermore, the author has constructed both models from the ground up, that is, starting from the mathematical principles and then translating these principles into code. Therefore, they do not rely on the external environment when used, but require the matching of the data.

### Code composition

The core code is divided into three main parts:

**Part 1: Correlation Analysis**  
This includes Pearson correlation coefficients and time-fixed effects. The corresponding script is `Correlation Analysis.py`.

**Part 2: Causality Analysis**  
This involves pre-matching, matching, and post-matching parts. The pre-matching analysis is found in `Pre-models.py`, while the matching and post-matching parts are in `Treatment Effect.py`.

**Part 3: Environmental and Economic Benefit Assessment**  
The corresponding script for this assessment is `Environmental Economic Assessment.py`.

This framework can be flexibly applied to cities with similar types of data, yielding a combined correlation and causality-based policy evaluation result.

**Note:** The relative paths referenced in the code files may not align with the current folder structure. Please contact the author if you require the complete directory structure or additional assistance. Welcome to leave your comments!
