# Housing Price Prediction Model — implemented from scratch using Matrices


## What is this project?

Basically, I wanted to see if I can implement Multiple Linear Regression completely from scratch using just Matrices. No fancy AI libraries like scikit-learn, nothing. Just pure python code and math to understand what is actually happening inside the algorithm.

I used four datasets originally, but now kept only the main ones: USA Housing (fake data) and King County (real data).

## How it works ?

1.  **Gradient Descent**: First, the code makes a random guess. Then it checks how much error is there, and tries to reduce it. It keeps doing this loop again and again until the error is minimal.
2.  **Learning Rate Decay**: In the starting, the code makes big jumps to find the answer. As it gets closer, it makes smaller steps so it doesn't overshoot the target.
3.  **Data Binning (Market Tiers)**: I split the houses into three groups: **Cheap (Low Tier)**, **Middle (Mid Tier)**, and **Expensive (High Tier)**. Why? Because cheap houses and expensive houses usually follow different logic.
4.  **Dashboards**: Every time I run the code, it generates a full dashboard showing how it performed.

---

## 1. USA Housing Data (Synthetic Control)

This dataset is fully fake (synthetic). I am using this just to check if our math logic is correct or not. Since the data is perfect, the graph should also come perfect.

### Full Market
Computer did a great job here. It found that **Income** is the main thing affecting the price.
*   **R** (Success Score): 0.9566 (Too good actually)
*   **R-Squared**: 0.9151
<details>
<summary>Click to see full model metrics</summary>

```text
Timestamp: 2026-01-18 06:06:03.028551
Dataset: USA_Housing.csv
Bin: Full_Data
------------------------------
R-Value: 0.9566
R-Squared: 0.9151
Final MSE: 0.085009
Best Epoch: 264

Feature Importances (Standardized Beta Weights):
  Area Population: 0.4396
  Avg. Area House Age: 0.4758
  Avg. Area Income: 0.6582
  Avg. Area Number of Bedrooms: 0.0199
  Avg. Area Number of Rooms: 0.3359
```
</details>

![USA Full](USA_Housing/Dashboard_Full_Data_2026-01-18_06-06-03.png)

### Low Tier (Cheap Houses)
*   **Success Score**: 0.8437
*   **What mattered**: Income and Population.
<details>
<summary>Click to see full model metrics</summary>

```text
Timestamp: 2026-01-18 06:05:57.719644
Dataset: USA_Housing.csv
Bin: Low-Tier
------------------------------
R-Value: 0.8437
R-Squared: 0.7118
Final MSE: 0.289728
Best Epoch: 325

Feature Importances (Standardized Beta Weights):
  Area Population: 0.6774
  Avg. Area House Age: 0.7362
  Avg. Area Income: 0.9176
  Avg. Area Number of Bedrooms: 0.0188
  Avg. Area Number of Rooms: 0.5357
```
</details>

![USA Low](USA_Housing/Dashboard_Low-Tier_2026-01-18_06-05-57.png)

### Mid Tier (Average Houses)
Here the computer struggled little bit.
*   **Success Score**: 0.6376
*   **What mattered**: Income only.
<details>
<summary>Click to see full model metrics</summary>

```text
Timestamp: 2026-01-18 06:05:59.499714
Dataset: USA_Housing.csv
Bin: Mid-Tier
------------------------------
R-Value: 0.6376
R-Squared: 0.4066
Final MSE: 0.595549
Best Epoch: 475

Feature Importances (Standardized Beta Weights):
  Area Population: 0.7712
  Avg. Area House Age: 0.8228
  Avg. Area Income: 0.9791
  Avg. Area Number of Bedrooms: 0.0035
  Avg. Area Number of Rooms: 0.6476
```
</details>

![USA Mid](USA_Housing/Dashboard_Mid-Tier_2026-01-18_06-05-59.png)

### High Tier (Expensive Houses)
*   **R** (Success Score): 0.8735
*   **What mattered**: Income and House Age.
<details>
<summary>Click to see full model metrics</summary>

```text
Timestamp: 2026-01-18 06:06:01.265477
Dataset: USA_Housing.csv
Bin: High-Tier
------------------------------
R-Value: 0.8735
R-Squared: 0.7630
Final MSE: 0.237040
Best Epoch: 599

Feature Importances (Standardized Beta Weights):
  Area Population: 0.6276
  Avg. Area House Age: 0.6724
  Avg. Area Income: 0.8668
  Avg. Area Number of Bedrooms: -0.0013
  Avg. Area Number of Rooms: 0.5208
```
</details>

![USA High](USA_Housing/Dashboard_High-Tier_2026-01-18_06-06-01.png)

#### Market Logic: Perfect Control
See the dashboards, the dots are following the red line perfectly. This proves that my matrix math code is solid and the Gradient Descent is working properly without any issues.

---

## 2. King County Data (Seattle area)

Now this is the real-world data, so obviously it is bit messy.

### Full Market
Location and "Grade" (quality) are the main things here.
*   **R** (Success Score): 0.8287
*   **R-Squared**: 0.6868
<details>
<summary>Click to see full model metrics</summary>

```text
Timestamp: 2026-01-18 06:06:11.668267
Dataset: kc_house_data.csv
Bin: Full_Data
------------------------------
R-Value: 0.8287
R-Squared: 0.6868
Final MSE: 0.313238
Best Epoch: 1442

Feature Importances (Standardized Beta Weights):
  bathrooms: 0.0686
  bedrooms: -0.0410
  condition: 0.0865
  floors: 0.0643
  grade: 0.3969
  lat: 0.3651
  long: 0.0105
  sqft_above: 0.3663
  sqft_basement: 0.1827
  sqft_living: -0.0263
  sqft_lot: 0.0403
  sqft_lot15: -0.0134
  view: 0.1046
  waterfront: 0.0298
  yr_built: -0.2334
  yr_renovated: 0.0224
```
</details>

![KC Full](kc_house_data/Dashboard_Full_Data_2026-01-18_06-06-11.png)

### Low Tier (Cheap Houses)
*   **Success Score**: 0.6542
*   **What mattered**: **Latitude** (Location). Basically where the house is, that decides the rate.
<details>
<summary>Click to see full model metrics</summary>

```text
Timestamp: 2026-01-18 06:06:05.119169
Dataset: kc_house_data.csv
Bin: Low-Tier
------------------------------
R-Value: 0.6542
R-Squared: 0.4279
Final MSE: 0.572057
Best Epoch: 1377

Feature Importances (Standardized Beta Weights):
  bathrooms: 0.1221
  bedrooms: -0.0590
  condition: 0.1107
  floors: 0.0215
  grade: 0.1940
  lat: 0.5230
  long: 0.0335
  sqft_above: 0.3935
  sqft_basement: 0.2165
  sqft_living: -0.0264
  sqft_lot: 0.0331
  sqft_lot15: 0.0284
  view: 0.0790
  waterfront: 0.0215
  yr_built: 0.0748
  yr_renovated: 0.0219
```
</details>

![KC Low](kc_house_data/Dashboard_Low-Tier_2026-01-18_06-06-05.png)

### Mid Tier (Average Houses)
This is the hardest part to guess!
*   **Success Score**: 0.3858
*   **What mattered**: Size (Sqft) and Location.
<details>
<summary>Click to see full model metrics</summary>

```text
Timestamp: 2026-01-18 06:06:07.419962
Dataset: kc_house_data.csv
Bin: Mid-Tier
------------------------------
R-Value: 0.3858
R-Squared: 0.1488
Final MSE: 0.851160
Best Epoch: 3625

Feature Importances (Standardized Beta Weights):
  bathrooms: 0.0924
  bedrooms: -0.0833
  condition: 0.0231
  floors: 0.1023
  grade: 0.2589
  lat: 0.2992
  long: 0.0850
  sqft_above: 0.3181
  sqft_basement: 0.2348
  sqft_living: -0.1370
  sqft_lot: 0.0385
  sqft_lot15: 0.0114
  view: 0.0616
  waterfront: 0.0120
  yr_built: -0.2912
  yr_renovated: -0.0099
```
</details>

![KC Mid](kc_house_data/Dashboard_Mid-Tier_2026-01-18_06-06-07.png)

### High Tier (Expensive Houses)
*   **Success Score**: 0.7653
*   **What mattered**: Size and Grade (Luxury features).
<details>
<summary>Click to see full model metrics</summary>

```text
Timestamp: 2026-01-18 06:06:09.423899
Dataset: kc_house_data.csv
Bin: High-Tier
------------------------------
R-Value: 0.7653
R-Squared: 0.5856
Final MSE: 0.414381
Best Epoch: 626

Feature Importances (Standardized Beta Weights):
  bathrooms: 0.0483
  bedrooms: -0.0418
  condition: 0.0770
  floors: -0.0488
  grade: 0.3164
  lat: 0.0983
  long: -0.1785
  sqft_above: 0.4513
  sqft_basement: 0.2004
  sqft_living: 0.0398
  sqft_lot: 0.0299
  sqft_lot15: -0.0532
  view: 0.0844
  waterfront: 0.1978
  yr_built: -0.1288
  yr_renovated: 0.0504
```
</details>

![KC High](kc_house_data/Dashboard_High-Tier_2026-01-18_06-06-09.png)

#### Market Logic: The "Funnel" of Uncertainty
If you check the **Residual Density** plot for King County, shape is like a funnel. As price goes up, errors are also increasing. Basically linear math is good for cheap homes but it gets confused with luxury houses. Also one big thing: Cheap houses depend on **Latitude** (Location meaning), but expensive houses depend on **Grade** (Quality).

---


## Summary of Results

Here is a quick summary of how the code performed:

| Dataset | Market Tier | Success Score (R) | Accuracy ($R^2$) | Biggest Factor |
| :--- | :--- | :--- | :--- | :--- |
| **King County** | Full | 0.8287 | 0.6868 | Grade / Size |
| | Low | 0.6542 | 0.4279 | Location (Lat) |
| | Mid | 0.3858 | 0.1488 | Size |
| | High | 0.7653 | 0.5856 | Size / Quality |


---

## Datasets Used

I used these datasets to check the code:

1. **USA Housing**: [USA Housing Dataset (Kaggle)](https://www.kaggle.com/datasets/kanths028/usa-housing)
2. **King County**: [House Sales in King County, USA (Kaggle)](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)


---

## Conclusion: A Lesson in Underfitting

So finally, what I learned from this project?

### 1. The "Middle-Class" Predictability Gap
Average houses (Mid-Tier) were the hardest to predict.
*   Cheap houses follow simple rules.
*   Expensive houses follow simple rules.
*   **Middle houses are chaotic.** Buyers in this range are very picky and have too many choices, so their behavior is not linear at all. Code basically struggled here.

**Final Verdict:** The code works perfectly on fake data, so math is correct. But **Real-World Housing is NOT Linear.** To get better accuracy, I need to use some advanced math that can understand "market soul", because simple lines cannot explain real estate market completely.
