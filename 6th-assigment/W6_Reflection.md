# Week 6 — Reflection: One Decision the Fully Bayesian Answer Would Have Changed

**Assignment:** *Probabilistic Models & Bayesian Inference* (Fusemachines AI Fellowship)
**Author:** Abhishek Shah
**Date:** 2026-06-06

---

## The decision

Imagine the VP of Retention asks me a sharp operational question:

> *"Our new contract tier has a churn-rate hypothesis ceiling of 25%. As soon as you can convince me the true rate is above 25%, I will sign off on a retention campaign. How many of these new customers do I need to observe before you tell me 'yes'?"*

This is exactly the situation Part 2 of the assignment was built around — sequential evidence on a small, never-before-seen segment.

## What the frequentist toolkit would have said

The standard frequentist answer is the one-proportion z-test sample-size formula:

$$n \;=\; \left\lceil \frac{\left(z_{\alpha/2}\sqrt{p_0(1-p_0)} + z_{\beta}\sqrt{p_1(1-p_1)}\right)^2}{(p_1 - p_0)^2}\right\rceil$$

Plugging in $p_0 = 0.25$ (the null), $p_1 = 0.265$ (the Telco base rate as the alternative), $\alpha = 0.05$, power $= 0.80$:

$$n_\text{freq} \;=\; \mathbf{6{,}304}$$

I would have walked back to the VP and told her: *"You need to wait until you have 6,304 customers on the new tier before I can give you a 5%-significance answer — the effect size is just too small for a smaller sample to be conclusive."*

## What the fully Bayesian answer said instead

In Q6 of the notebook I ran sequential Bayesian updating on the Telco churn outcomes starting from a Beta(2, 8) prior, computing $P(\theta > 0.25 \mid \text{data})$ at each step via 10,000 Monte-Carlo samples from the running posterior.

**The posterior first crosses the 90% threshold at $n = 17$ customers.**

That is a 370× reduction in the sample size required to act. The retention campaign — assuming it has positive expected ROI — would have started **6,287 customers' worth of waiting** earlier under the Bayesian framing.

## The mechanism — *why* the gap is so enormous

This is not the Bayesian framework being "lucky" or the prior carrying the load. Three distinct mechanisms compound:

1. **Sequential updating without multiple-testing penalty.** The frequentist z-test fixes a sample size in advance because peeking at the data and stopping early inflates the false-positive rate. In the Bayesian update, every observation is a literal application of Bayes' rule on the current posterior — the framework is *built* for sequential decisions, no peeking penalty applies, and $P(\theta > 0.25)$ is always a coherent statement at every $n$.

2. **The right question is being asked.** The frequentist computes $P(\text{data as extreme as observed} \mid H_0: \theta = 0.25)$ — a statement about the procedure under a counterfactual world where $\theta$ is exactly 0.25. The Bayesian computes $P(\theta > 0.25 \mid \text{data})$ — a direct statement about the quantity the VP cares about. Different questions naturally require different evidence to answer.

3. **The prior is informative, not decorative.** Beta(2, 8) encodes our domain belief that most contract segments churn below 30%. With only 40–50 observations the prior contributes meaningfully — it does *not* dominate (the prior pull on Group A_small was only **0.0417**, MLE = 0.3750 vs MAP = 0.3333) but it provides a starting point that the frequentist test, having no prior, cannot use.

## The cross-check that confirms this is not pathology

The same notebook contains the convergence sanity check: in Part 6 the Bayesian 94% HDI for $\beta_\text{Month-to-month}$ is **[1.086, 1.496]** with posterior mean **1.294**, while the frequentist MLE (`sklearn.LogisticRegression(C=1e6)`) is **1.293**. With ~5,000 training observations the two approaches agree to three decimal places — exactly as theory predicts when the data dominates the prior. The framework does not give a different answer when the data is abundant; it only gives a *better* answer when the data is scarce, which is precisely the regime Part 2 was built around.

## Why this matters for the VP's actual question

The VP's two original concerns at the top of the project guide were:

1. *"What is the margin of error on this 73% churn probability?"*
2. *"Your model has never seen this 40-customer segment — how far should I trust it?"*

The Bayesian framework answers both **with the same machinery**: the posterior distribution. The point of this assignment is that "a number with no distribution attached" is an incomplete answer to either question, and the cost of that incompleteness is measured here in **6,287 customers of unnecessary delay** before a profitable retention campaign could launch.

---

*Word count: ~580 — fits on a single page.*
