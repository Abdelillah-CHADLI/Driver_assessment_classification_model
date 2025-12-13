# Driving Telemetry — Cleaning, EDA, and Risk Assessment (Safe/Moderate/Dangerous)

## Project Overview
This project analyzes vehicle sensor telemetry collected from real-world driving to support building a **driving assessment** model that categorizes driving behavior into **safe**, **moderate**, and **dangerous**. The dataset used in this workspace originally comes from driver-profiling / driver-identification research (10 drivers labeled **A–J**). In this project, that driver identifier is treated as an ID (not the target label) and is mainly used for exploratory analysis and later validation strategies.

## Goals
- Produce a **single, unified cleaned dataset** that the entire team can use consistently.
- Perform **EDA (Exploratory Data Analysis)** to understand distributions, outliers, and correlations in key driving-behavior signals.
- Prepare the data and documentation needed to proceed to later phases (labeling strategy + modeling).

## Dataset Summary
- Input file: `data.csv`
- Row count: ~94k observations
- Feature types: mostly numeric sensors (speed, acceleration, steering, braking, engine signals)
- Identifier: `Class` contains driver IDs (A–J)

## Work Completed in This Workspace
### Phase 1 — Feature Screening (Report)
- See: `Data_Mining_Project_Phase1.pdf`
- Outcome: documented which features were considered relevant vs. irrelevant (e.g., dropping ECU tuning/maintenance-like signals that do not reflect driving behavior).

### Phase 2 — Cleaning & EDA (Notebook)
- See: `cleaning.ipynb`
- Cleaning tasks covered:
  - Validate missing values
  - Detect and remove **true** duplicates (full-row duplicates only)
  - Standardize basic formatting (e.g., trimming column headers; consistent driver ID formatting)
  - Remove clearly non-informative columns (e.g., constant features)
  - Export a standardized dataset for the team
- Output file: `cleaned_data.csv`

## Key Notes / Assumptions
- **Driver ID is not the target.** `Class` identifies who drove (A–J). It is not a “safe/dangerous” label.
- **Outliers can be meaningful.** In driving telemetry, extreme values may represent important safety events (hard braking, harsh cornering) and should not be removed blindly.
- **Deduplication must be careful.** Low-resolution time fields can cause accidental loss of valid rows if used as a uniqueness key.

## How to Use
1. Open `cleaning.ipynb` and run cells top-to-bottom.
2. Confirm the exported dataset `cleaned_data.csv` is created/updated.
3. Use `cleaned_data.csv` for team EDA and for later phases.

## Next Phase (Planned)
- Define/obtain labels for safe/moderate/dangerous driving (ground truth or weak supervision rules).
- Move from row-level readings to window/segment features (e.g., rolling statistics) for a robust driving assessment model.
