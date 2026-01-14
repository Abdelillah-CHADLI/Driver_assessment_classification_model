# Driving Behavior Clustering Analysis

**Component:** Unsupervised Clustering for Driver Behavior Profiling  
**Course Methodologies:** Hierarchical Clustering, K-means, DBSCAN  

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Methodology](#methodology)
3. [Notebook Structure](#notebook-structure)
6. [Output Interpretation](#output-interpretation)
7. [Integration with Anomaly Detection](#integration-with-anomaly-detection)
8. [Limitations & Assumptions](#limitations--assumptions)

---

## Project Overview

### **Objective**
Identify distinct driving behavior profiles using unsupervised clustering techniques to support the assessment of cautious vs. dangerous driving patterns.

### **Problem Context**
- **Unsupervised Learning Task:** No labeled data for "cautious" or "dangerous" driving
- **Feature-Blind Approach:** 28 numeric features used without domain-specific interpretation
- **Fixed Time Windows:** Each row represents driving behavior over a fixed time interval
- **Complementary Analysis:** Works alongside anomaly detection (handled by another team member)

### **Key Assumptions**
1. Most drivers exhibit cautious behavior (majority class)
2. Different driving styles can be distinguished by feature patterns
3. Cluster structure exists in the feature space
4. Distance-based similarity is meaningful for driving behavior

---

## Methodology

### **Three Clustering Algorithms Applied**

| Algorithm | Type | Strengths | When to Use |
|-----------|------|-----------|-------------|
| **Hierarchical** | Agglomerative | Reveals nested structure, no need to pre-specify k | Exploratory analysis, small-medium datasets |
| **K-means** | Partitioning | Fast, scalable, interpretable centroids | Spherical clusters, known approximate k |
| **DBSCAN** | Density-based | Arbitrary shapes, noise detection | Non-spherical clusters, outlier identification |

### **Evaluation Metrics** (Internal Validation)

Since this is unsupervised with no ground truth, we use:

- **Silhouette Score** (range: -1 to 1)
  - Measures how similar samples are to their own cluster vs. other clusters
  - Higher is better (>0.5 = good, 0.3-0.5 = moderate, <0.3 = weak)
  
- **Calinski-Harabasz Index** (range: 0 to ∞)
  - Ratio of between-cluster to within-cluster variance
  - Higher is better (more separated, compact clusters)
  
- **Davies-Bouldin Index** (range: 0 to ∞)
  - Average similarity between each cluster and its most similar one
  - Lower is better (more distinct clusters)

---

## Notebook Structure

### **Section 1: Data Loading & Exploration**
- Dataset dimensions and feature count
- Time window coverage
- Initial data quality checks

### **Section 2: Feature Preprocessing**
- **Feature Selection:** Exclude metadata (`driver_id`, `pathorder`, `time_s`)
- **Missing Value Handling:** Forward/backward fill if needed
- **Standardization:** Z-score normalization (mean=0, std=1)
  - **Why:** Distance-based algorithms require features on same scale

### **Section 3: Dimensionality Reduction (PCA)**
- Reduce to 2D for visualization
- Shows data distribution in principal component space
- Reports variance explained by first 2 components

### **Section 4: Hierarchical Clustering**
- **Distance Matrix Demo:** 10×10 sample showing pairwise Euclidean distances
- **Linkage Methods Tested:**
  - Single (MIN): Minimum distance between clusters
  - Complete (MAX): Maximum distance between clusters
  - Average (UPGMA): Average distance between all pairs
- **Dendrograms:** Visual representation of cluster hierarchy
- **Optimal k Selection:** Tested k=2,3,4,5,6,8,10
- **Cluster Centroids:** Prototype feature vectors for each cluster

**Output:**
- 3 dendrograms (one per linkage method)
- Best configuration based on silhouette score
- Cluster visualization in PCA space
- Centroid table (first 5 features shown)

### **Section 5: K-means Clustering**
- **Elbow Method:** Plot inertia (within-cluster sum of squares) vs. k
- **Silhouette Analysis:** Plot silhouette score vs. k
- **Additional Metrics:** Calinski-Harabasz and Davies-Bouldin indices
- **Optimal k:** Selected as k with highest silhouette score
- **Cluster Centroids:** Mean feature vectors + dominant features

**Output:**
- 4 evaluation plots (elbow, silhouette, CH, DB)
- Optimal k recommendation
- Cluster size distribution
- Centroid interpretation table
- PCA visualization with cluster colors

### **Section 6: DBSCAN Clustering**
- **k-Distance Plot:** Used to identify optimal `eps` (epsilon)
- **Parameter Grid Search:**
  - `eps`: Tested 5 candidate values (percentiles + statistics)
  - `min_samples`: Tested 3, 5, 10
- **Selection Criteria:** Maximize silhouette while keeping noise < 50%
- **Cluster Prototypes:** Centroids for each dense cluster (noise excluded)

**Output:**
- k-distance plot with suggested eps
- Configuration results table (15 rows)
- Best configuration with reasoning
- Noise point percentage
- PCA visualization with noise marked as -1

### **Section 7: Comparative Analysis**
- Side-by-side table comparing all three methods
- Best method recommendation
- Cluster count comparison

### **Section 8: Interpretation & Recommendations**
- Cluster quality assessment
- Integration guidance with anomaly detection
- Practical usage instructions
- Next steps for validation

---

### **Data **

  - `driver_id`: String/categorical (metadata only)
  - `pathorder`: Integer (metadata only)
  - `time_s`: Integer (metadata only)

---

## Output Interpretation

### **Understanding Cluster Counts**

| Scenario | Interpretation | Action |
|----------|----------------|--------|
| **2-3 clusters** | Simple bimodal/trimodal behavior | May be too coarse; investigate k=4-6 |
| **4-6 clusters** | Diverse driving profiles | Good balance; analyze centroids |
| **7+ clusters** | High fragmentation | May be overfitting; consider dimensionality reduction |

### **Understanding Silhouette Scores**

| Score Range | Interpretation | Decision |
|-------------|----------------|----------|
| **> 0.5** | Good separation | Clusters are distinct |
| **0.3 - 0.5** | Moderate separation | Acceptable; some overlap |
| **< 0.3** | Weak separation | Consider fewer clusters or DBSCAN |

---

## Limitations & Assumptions

### **Known Limitations**

1. **No Ground Truth Validation**
   - Cannot verify if clusters correspond to "cautious" vs. "dangerous"
   - Rely entirely on internal metrics and domain expert review

2. **Feature-Blind Approach**
   - Features treated equally without domain weighting
   - May miss important domain-specific patterns

3. **Euclidean Distance Assumption**
   - Assumes linear feature space
   - May not capture complex non-linear relationships

4. **Computational Constraints**
   - Hierarchical uses 5,000 sample subset (full dataset too large)
   - Full dataset used for K-means and DBSCAN

5. **Temporal Independence**
   - Each time window treated independently
   - Doesn't model sequential driving behavior

### **Methodological Assumptions**

- **Cluster-able Data:** Assumes natural groupings exist
- **Standardization Validity:** Assumes Z-score normalization is appropriate
- **Fixed Window Size:** Assumes time windows are meaningful units
- **Static Parameters:** Parameters chosen heuristically, not optimized globally


## Integration with Anomaly Detection

Workflow for Combined Analysis
1. Clustering Analysis (This Notebook)
   ↓
   Produces: Cluster labels for each time window
   ↓
2. Anomaly Detection (Teammate's Work)
   ↓
   Produces: Anomaly scores/labels for each time window
   ↓
3. Integration Analysis
   ↓
   Cross-tabulate: Cluster vs. Anomaly


## Notes for Anomaly Detection

- This notebook produces cluster labels you can use as features
- Consider: "Distance from cluster centroid" as an anomaly signal
- Small clusters (< 5% of data) warrant investigation