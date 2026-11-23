#### Key findings

1. **Every ensemble beats the single decision tree.** F1 rises from **0.535** (one tree) to **0.568** (Random Forest), and ROC-AUC from **0.843** to **≈ 0.863**. Averaging many trees removes most of a single tree's variance.
2. **The winner depends on the metric, and the margins are small:**
   * 🥇 **Random Forest (Bagging)** has the best **F1 (0.568)**, **accuracy (0.820)** and **precision (0.494)**.
   * **XGBoost, Gradient Boosting (Boosting) and the Hybrid RF + ANN** share the best **ROC-AUC (≈ 0.863)**, and the **Hybrid** has the best **PR-AUC (0.600)**.
   * The top eight models are within ≈ 0.02 F1 of each other, and all nine tree ensembles lie between 0.849 and 0.863 ROC-AUC. The ceiling is set mainly by the information in daily, ~50 km reanalysis data, not by the choice of ensemble.

#### Why did the Random Forest (Bagging) win on F1?

* **A noisy target rewards variance reduction.** Whether a town gets ≥ 10 mm tomorrow depends partly on short-lived convective storms that daily averages cannot see. From the features' point of view, many labels are effectively noise.
  * **Bagging averages this noise away.** It trains 300 de-correlated trees on bootstrap samples.
  * **Boosting keeps re-focusing on the hardest, often unpredictable days.** **AdaBoost**, which explicitly up-weights misclassified days, has the **lowest precision (0.407)** of all the ensembles.
* **It has a better-balanced operating point.** With balanced class weights, the boosted models (which minimise weighted log-loss) push many borderline days above the 0.5 threshold.
  * Boosting therefore gets **high recall (0.74–0.82) but more false alarms** (precision 0.41–0.45).
  * The Random Forest's probabilities are averages over 300 trees whose leaves hold at least 10 days each, so they are **smoother and more conservative**. At 0.5 it trades some recall (0.669) for many fewer false alarms, which gives the best F1.
  * On threshold-free ranking (ROC-AUC), boosting is marginally better (+0.002). Boosting's strength here is *ranking* days, not the default decision threshold.
* **Feature randomisation helps with correlated predictors.** Rain today, the 3- and 7-day totals, humidity, dew point and cloud cover are strongly correlated. `max_features="sqrt"` makes the trees use different feature subsets, which de-correlates their errors.
  * The **out-of-bag accuracy (0.851)** gave a free validation estimate.
  * Test accuracy is a little lower (0.820) because **2025–26 was wetter** than the training years (17.7% vs 15.3% heavy-rain days), which is a mild distribution shift.

#### Why didn't Stacking win?
The logistic-regression meta-learner put almost all its weight on **LightGBM (+6.83)**, with small *negative* weights on the **RF (−0.79)** and the **ANN (−0.23)**.
* The base models are **highly correlated** (RF vs ANN probabilities: r = 0.94), because all of them rely on the same dominant signal, today's rain. There was little complementary information left for the meta-learner to exploit.
* The extra layer added variance. Stacking pays off when base learners make **different** errors, and here they mostly agree.

#### Why did the Hybrid RF + ANN do well?
Averaging the conservative RF with the high-recall ANN produced a middle operating point (**F1 0.562, 2nd overall**). It also gave the **best PR-AUC (0.600)** and a **ROC-AUC (0.863) higher than either parent** (RF 0.862, ANN 0.857). Combining two *different model families* improved how heavy-rain days are ranked, which is exactly the motivation for hybridisation.

#### What drives the predictions? (feature importance)
* **All models agree that rain today is the strongest predictor.** Monsoon wet spells persist, so a wet day is usually followed by another.
* Next come the **3- and 7-day rainfall totals** (wet-spell memory), **seasonality** (day-of-year, monsoon season), **cloud cover / solar radiation** and the **1-day pressure change**. Falling pressure signals an approaching low-pressure system.
* The late-November 2025 cyclone (Ditwah) is an example. On 26 Nov 2025 the models gave Kandy an ~88–95% chance of heavy rain, and **196 mm** fell the next day.
* XGBoost also leans on **town, province and monsoon season**, learning region-specific corrections.

#### Regional insight (Random Forest)
* **By zone:** the **Dry zone** is the most predictable (ROC-AUC 0.882), because its heavy rain is concentrated in the NE monsoon and second inter-monsoon. The **Wet zone** is the hardest (0.834), because rain there is frequent and often convective.
* **By season:** the **second inter-monsoon (Oct–Nov)** has the highest F1 (0.739). The **first inter-monsoon and the NE monsoon** are hardest (F1 ≈ 0.45–0.47), because afternoon thunderstorms are hard to anticipate from daily data.

#### Which model would I deploy?
* **Flood early warning**, where a missed event is costly: **XGBoost / Gradient Boosting** (best ROC-AUC, recall ≈ 0.76–0.80), or the RF with a lower threshold.
* **Balanced warnings with fewer false alarms:** the **Random Forest**. It is fast to train (13 s), robust, and comes with a built-in OOB check.
