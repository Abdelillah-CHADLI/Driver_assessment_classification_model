# Driver Risk Classification Using Anomaly Detection
## Technical Report

---

## Executive Summary

This report presents a machine learning approach for classifying driver behavior based on risk levels using vehicle telemetry data. We employ **anomaly detection techniques** (Isolation Forest and One-Class SVM) to identify risky driving patterns from 10 drivers.

**Key Findings:**
- 10 drivers (A-J) analyzed across 94,380 sensor readings
- 4,997 time windows (5-second intervals) extracted
- **Driver A** identified as highest risk (28.6% risky windows)
- **Lateral acceleration** is the strongest predictor of risky driving
- Ensemble approach (consensus of both models) provides robust classification

---

## 1. Introduction

### 1.1 Problem Statement

Driver behavior assessment is critical for:
- Insurance risk pricing
- Fleet management safety
- Autonomous vehicle development
- Traffic safety research

The challenge is to automatically classify drivers into risk categories (High/Medium/Low) based solely on vehicle sensor data, without prior labeled examples of "risky" vs "safe" driving.

### 1.2 Dataset Overview

| Metric | Value |
|--------|-------|
| Total samples | 94,380 |
| Number of drivers | 10 (A-J) |
| Features | 31 sensor readings |
| Sampling rate | ~1 Hz |

**Key sensor features used:**
- Vehicle speed
- Longitudinal acceleration (braking/acceleration)
- Lateral acceleration (cornering)
- Engine speed (RPM)
- Throttle position
- Brake pressure (master cylinder)
- Steering wheel angle and speed
- Fuel consumption

---

## 2. Methodology

### 2.1 Why Anomaly Detection for Risk Classification?

The fundamental insight is that **risky driving is inherently anomalous**. Most driving is normal/safe, while dangerous behaviors (harsh braking, aggressive acceleration, sharp turns) are statistical outliers.

This makes anomaly detection ideal because:

1. **No labeled data required** - We don't need pre-labeled "risky" vs "safe" examples
2. **Natural fit** - Risky behaviors ARE anomalies by definition
3. **Adaptable threshold** - Can tune sensitivity based on business needs
4. **Handles novel risks** - Detects unusual patterns not explicitly programmed

### 2.2 Feature Engineering: 5-Second Time Windows

Raw sensor data is aggregated into 5-second windows to capture driving behavior patterns:

```
For each 5-second window, we extract:
├── Mean      (average behavior)
├── Std       (variability/consistency)
├── Max       (peak intensity)
├── Min       (baseline)
├── Range     (max - min spread)
└── Skewness  (asymmetry of behavior)
```

**Why 5 seconds?**
- Short enough to capture individual maneuvers (lane change, braking event)
- Long enough to filter sensor noise
- Aligns with research standards in driver behavior analysis
- Produces ~500 windows per driver for robust statistics

**Total features per window:** 11 sensors × 6 statistics = **66 features**

### 2.3 Risk Score Calculation

A composite risk score combines multiple aggressive driving indicators:

| Risk Component | Based On | Rationale |
|----------------|----------|-----------|
| Speed Risk | Speed variability (std) | Erratic speed changes |
| Acceleration Risk | Longitudinal accel std | Harsh accel/braking |
| Lateral Risk | Lateral accel std | Aggressive cornering |
| Brake Risk | Max brake pressure | Emergency braking |
| Steering Risk | Steering speed std | Jerky steering |
| Throttle Risk | Max throttle | Aggressive acceleration |

Scores are normalized using **RobustScaler** (resistant to outliers) and combined into a 0-1 composite score.

---

## 3. Anomaly Detection Models

### 3.1 Isolation Forest

**How it works:**

Isolation Forest isolates anomalies by randomly partitioning data. The key insight is that **anomalies are easier to isolate** than normal points.

```
Algorithm:
1. Randomly select a feature
2. Randomly select a split value between min and max
3. Repeat recursively until point is isolated
4. Anomalies require fewer splits (shorter path length)
```

**Why suitable for driver risk:**
- Handles high-dimensional data (66 features) efficiently
- No assumption about data distribution
- Naturally handles mixed feature types
- Computationally efficient O(n log n)
- Robust to irrelevant features

**Parameters used:**
```python
IsolationForest(
    n_estimators=200,      # Number of trees
    contamination=0.15,    # Expected proportion of anomalies
    random_state=42
)
```

### 3.2 One-Class SVM

**How it works:**

One-Class SVM learns a boundary around "normal" data in high-dimensional space. Points outside this boundary are anomalies.

```
Concept:
- Map data to high-dimensional space using RBF kernel
- Find the smallest hypersphere containing most points
- Points outside = anomalies
```

**Why suitable for driver risk:**
- Effective for complex, non-linear boundaries
- Works well with standardized features
- Mathematically principled (maximum margin)
- Complementary to Isolation Forest (different approach)

**Parameters used:**
```python
OneClassSVM(
    kernel='rbf',      # Radial Basis Function kernel
    gamma='scale',     # Automatic gamma based on features
    nu=0.15            # Upper bound on anomaly fraction
)
```

### 3.3 Ensemble Approach

We combine both models for robust classification:

| Strategy | Definition | Use Case |
|----------|------------|----------|
| **Ensemble (OR)** | Either model flags as risky | High sensitivity, catch all risks |
| **Consensus (AND)** | Both models agree | High precision, confident predictions |

**Results:**
- Ensemble: 18.6% flagged as risky
- Consensus: 11.5% flagged as risky (used for final classification)

---

## 4. Results

### 4.1 Overall Detection Performance

| Model | Risky Windows | Anomaly Rate |
|-------|---------------|--------------|
| Isolation Forest | 750 | 15.0% |
| One-Class SVM | 752 | 15.0% |
| Consensus (both agree) | 573 | 11.5% |

### 4.2 Driver Risk Classification

Based on consensus risky window ratio:

| Risk Level | Threshold | Drivers |
|------------|-----------|---------|
| High risk | >=30% risky windows | None |
| Medium risk | 15-30% risky windows | A (28.6%), C (15.8%) |
| Low risk | <15% risky windows | B, D, E, F, G, H, I, J |

**Driver Risk Summary Table:**

| Driver | Mean Risk Score | Risky Ratio (IF) | Risky Ratio (SVM) | Consensus | Level |
|--------|-----------------|------------------|-------------------|-----------|-------|
| A | 0.145 | 33.7% | 33.3% | 28.6% | Medium |
| C | 0.141 | 20.0% | 21.3% | 15.8% | Medium |
| I | 0.122 | 13.6% | 13.2% | 9.9% | Low |
| D | 0.120 | 8.9% | 10.6% | 6.5% | Low |
| B | 0.117 | 17.0% | 14.5% | 10.2% | Low |
| J | 0.114 | 14.7% | 13.2% | 9.3% | Low |
| E | 0.112 | 13.3% | 13.5% | 8.6% | Low |
| H | 0.111 | 13.0% | 13.2% | 8.3% | Low |
| F | 0.109 | 10.4% | 14.5% | 7.7% | Low |
| G | 0.102 | 13.5% | 17.2% | 9.3% | Low |

### 4.3 Feature Importance

Top features distinguishing risky from normal driving:

| Rank | Feature | Importance Score |
|------|---------|------------------|
| 1 | **Lateral acceleration (mean)** | 24.8 |
| 2 | Vehicle speed (skewness) | 1.8 |
| 3 | Lateral acceleration (min) | 1.0 |
| 4 | Lateral acceleration (std) | 0.9 |
| 5 | Lateral acceleration (range) | 0.8 |

**Key insight:** Lateral acceleration dominates risk prediction, indicating that **aggressive cornering and lane changes** are the primary indicators of risky driving.

### 4.4 PCA Visualization

Principal Component Analysis reveals clear separation between normal and risky driving patterns:

- **PC1 (22.9%):** Primarily captures speed and acceleration magnitude
- **PC2 (13.2%):** Captures steering and lateral dynamics
- **Total explained variance:** 36.1%

The PCA plot shows risky windows (red) clustered at the periphery of the normal driving distribution (green), validating the anomaly detection approach.

---

## 5. Potential Improvements

### 5.1 Short-term Improvements

1. **Add temporal features**
   - Include time-lagged features (previous window statistics)
   - Capture driving pattern trends over longer periods

2. **Context-aware risk scoring**
   - Weight features by driving context (highway vs urban)
   - Use road gradient information

3. **Ensemble with more models**
   - Add Local Outlier Factor (LOF)
   - Add Autoencoder-based anomaly detection

### 5.2 Medium-term Improvements

4. **Deep Learning approaches**
   - LSTM for sequence modeling
   - Autoencoder for unsupervised anomaly detection
   - Transformer-based models for long-range dependencies

5. **Semi-supervised learning**
   - Use small labeled dataset to calibrate thresholds
   - Active learning to improve with expert feedback

6. **Multi-task learning**
   - Jointly predict risk level AND driver identity
   - Share representations between tasks

### 5.3 Long-term Improvements

7. **Real-time deployment**
   - Edge computing for in-vehicle processing
   - Streaming anomaly detection algorithms

8. **Explainable AI**
   - SHAP values for individual predictions
   - Natural language explanations for drivers

9. **Transfer learning**
   - Pre-train on large driving datasets
   - Fine-tune for specific fleets/regions

---

## 6. Conclusion

### 6.1 Key Takeaways

1. **Anomaly detection is well-suited for driver risk classification** because risky behaviors are inherently statistical outliers

2. **Isolation Forest and One-Class SVM provide complementary perspectives** - using consensus improves precision

3. **Lateral acceleration is the dominant risk indicator** - aggressive cornering/lane changes are the clearest signal

4. **Driver A shows consistently elevated risk** across all metrics (28.6% risky windows)

5. **The ensemble approach provides robust results** - both models agree on the most clearly risky patterns

### 6.2 Recommendations

For a production system, we recommend:

1. Deploy anomaly detection for **real-time risk scoring**
2. Use the **consensus approach** for high-confidence alerts
3. Use the **ensemble approach** for comprehensive monitoring
4. Implement **feedback loops** to continuously improve models

---

## Appendix A: Technical Implementation Details

### A.1 Libraries Used

```python
pandas==2.x          # Data manipulation
numpy==1.x           # Numerical operations
scikit-learn==1.x    # ML models
scipy==1.x           # Statistical functions
matplotlib==3.x      # Visualization
seaborn==0.x         # Statistical visualization
```

### A.2 Reproducibility

- Random seed: 42
- Train/test split: Not applicable (unsupervised)
- Cross-validation: Not applicable (anomaly detection)

### A.3 Computational Performance

| Operation | Time | Memory |
|-----------|------|--------|
| Data loading | <1s | ~50MB |
| Window creation | ~60s | ~100MB |
| Isolation Forest | <1s | ~10MB |
| One-Class SVM | ~2s | ~20MB |
| Total pipeline | ~65s | ~150MB |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Anomaly** | Data point significantly different from majority |
| **Contamination** | Expected proportion of anomalies in data |
| **Isolation Forest** | Tree-based anomaly detection algorithm |
| **One-Class SVM** | Support Vector Machine trained on single class |
| **RBF Kernel** | Radial Basis Function, maps data to infinite dimensions |
| **Time Window** | Fixed-duration segment of sensor data |
| **RobustScaler** | Scaling method resistant to outliers (uses median/IQR) |

---

*Report generated: February 4, 2026*  
*Project: Driver Assessment Classification Model*
