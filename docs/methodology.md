# Statistical methodology

## 1. Estimand

The simulation engine evaluates the **sample average treatment effect (ATE)** implied by the known structural model for each generated covariate matrix. This choice is exact and especially useful when the DGP includes heterogeneous treatment effects or binary outcomes.

For continuous outcomes with structural potential-outcome means

$
\mu_0(X)=\beta_0+\beta'X+f(X),\qquad
\mu_1(X)=\mu_0(X)+\tau(X),
$

the replication-specific truth is

$
\theta=\frac{1}{n}\sum_{i=1}^n\{\mu_1(X_i)-\mu_0(X_i)\}.
$

When \(\tau(X)=\tau_0\), this is exactly \(\tau_0\) in every replication. When \(\tau(X)=\tau_0+\tau_1X_1\), the sample ATE varies with the generated covariate sample and is recorded explicitly.

For binary outcomes,

$
p_a(X)=\mathrm{logit}^{-1}\{\beta_0+\beta'X+f(X)+a\tau(X)\},
$

and the target is the sample-average risk difference

$
\theta=\frac{1}{n}\sum_i\{p_1(X_i)-p_0(X_i)\}.
$

Thus `tau0` is a structural log-odds coefficient in binary DGPs, while the evaluated causal estimand is a risk difference.

## 2. Covariates

The generator supports one to ten covariates:

- iid standard Normal;
- Uniform on \([-\sqrt 3,\sqrt 3]\), giving variance one;
- Bernoulli(0.5);
- equicorrelated multivariate Normal with unit variances.

The baseline linear coefficient vector decreases with dimension as

$
\beta_j = \text{beta\_scale}/\sqrt{j},
$

which keeps high-dimensional preset signal magnitudes numerically reasonable.

## 3. Treatment assignment

### Randomized

$
A\sim\mathrm{Bernoulli}(p),\qquad A\perp X.
$

### Confounded logistic assignment

$
P(A=1\mid X)=\mathrm{logit}^{-1}(\alpha_0+\alpha_1X_1+0.35\alpha_1X_2),
$

with the `X2` term omitted when only one covariate exists. `confounding_strength` is \(\alpha_1\). The true propensity can optionally be clipped at an explicitly configured floor for numerical scenario control. Poor-overlap presets use strong treatment selection and a very small floor so that extreme propensities remain visible rather than hidden.

## 4. Outcome surfaces

The continuous-outcome family starts from

$
Y=\beta_0+\beta'X + A\tau(X)+\varepsilon.
$

Optional structures include:

- quadratic: an added centered \(qX_1^2\) term;
- nonlinear: an added \(c\sin(X_1)\), plus a centered quadratic term in `X2` when available;
- heterogeneous effect: \(\tau(X)=\tau_0+\tau_1X_1\);
- interaction: an additional \(A\gamma X_1\) contribution.

The same systematic predictor enters the binary-outcome logit model.

## 5. Error distributions

For continuous outcomes:

- **Gaussian:** \(\varepsilon\sim N(0,\sigma^2)\).
- **Student-t:** draws are rescaled so their standard deviation is approximately the requested `noise_sd`; degrees of freedom must exceed two.
- **Skewed:** a centered/scaled log-normal error with mean zero and requested SD.
- **Heteroskedastic:**

  $
  \mathrm{SD}(\varepsilon\mid X)
  =\sigma\sqrt{1+hX_1^2}.
  $

Binary outcomes do not add a separate continuous error term; Bernoulli variation is generated from the structural outcome probability.

## 6. Estimators

### Difference in means

$
\hat\theta=\bar Y_1-\bar Y_0.
$

The reported SE is the usual unequal-variance two-sample approximation

$
\widehat{SE}=\sqrt{s_1^2/n_1+s_0^2/n_0}.
$

This estimator is unbiased under randomized assignment for the mean treatment effect but generally biased when treatment is confounded with prognostic covariates.

### OLS regression adjustment

The fitted model is

$
Y_i=\beta_0+\theta A_i+\beta'X_i+u_i.
$

The treatment coefficient is reported with either conventional OLS covariance or HC3 heteroskedasticity-robust covariance. Under randomized treatment, regression adjustment remains a useful precision benchmark; under confounding, its causal validity depends on correct adjustment and model assumptions.

### Outcome regression / g-computation

A parametric regression is fit to `Y ~ A + X`, then every observation is predicted under `A=1` and `A=0`. The point estimate is

$
\hat\theta=\frac1n\sum_i\{\hat m_1(X_i)-\hat m_0(X_i)\}.
$

Continuous outcomes use OLS and binary outcomes use logistic GLM. Standard errors use a delta method based on the fitted coefficient covariance.

### Propensity-score model

Propensity estimators fit an unpenalized logistic MLE using a compact Newton/IRLS routine. The researcher can deliberately restrict the propensity covariates in the UI to study nuisance-model misspecification.

### IPW

The Horvitz-Thompson ATE is

$
\hat\theta_{IPW}=\frac1n\sum_i\left[
\frac{A_iY_i}{\hat e(X_i)}-
\frac{(1-A_i)Y_i}{1-\hat e(X_i)}
\right].
$

Its reported within-replication SE is the empirical SD of the corresponding influence contributions divided by \(\sqrt n\), treating the fitted propensity as fixed. This is a transparent approximation and is intentionally documented rather than presented as exact nuisance-estimation variance accounting.

### Stabilized IPW / Hájek estimator

The point estimate is the difference between normalized weighted means,

$
\hat\mu_1=
\frac{\sum_i A_iY_i/\hat e_i}{\sum_i A_i/\hat e_i},
\qquad
\hat\mu_0=
\frac{\sum_i (1-A_i)Y_i/(1-\hat e_i)}{\sum_i (1-A_i)/(1-\hat e_i)},
$

with \(\hat\theta=\hat\mu_1-\hat\mu_0\). The SE uses the corresponding ratio-estimator influence approximation.

### Doubly robust AIPW

The implementation uses deterministic two-fold cross-fitting. Propensity and treatment-specific outcome nuisance models are fit on the opposite fold. For observation \(i\),

$
\hat\psi_i=
\hat m_1(X_i)-\hat m_0(X_i)
+\frac{A_i\{Y_i-\hat m_1(X_i)\}}{\hat e(X_i)}
-\frac{(1-A_i)\{Y_i-\hat m_0(X_i)\}}{1-\hat e(X_i)}.
$

Then

$
\hat\theta_{AIPW}=n^{-1}\sum_i\hat\psi_i,
$

and the SE is the empirical SD of \(\hat\psi_i-\hat\theta\) divided by \(\sqrt n\). Under standard identification/regularity conditions, AIPW is consistent if either the propensity nuisance model or the outcome nuisance model is correctly specified. The simulation truth is a replication-specific sample ATE; this conventional influence-function SE is closer to a superpopulation-style large-sample approximation, so coverage under heterogeneous effects should be interpreted with that estimand/inference distinction in mind.

### Huber robust regression

The original project’s Huber M-estimator is retained for continuous-outcome robustness exploration. Its treatment coefficient and asymptotic statsmodels RLM SE are reported. It is not intended as a general binary-outcome causal estimator.

## 7. Confidence intervals and rejection

Estimator-specific intervals use the configured nominal confidence level and a standard-normal critical value unless the underlying statsmodels estimator directly supplies the equivalent asymptotic covariance. A null value of zero is used by the current calibration modes. An interval rejects the null when it lies entirely above or below zero.

## 8. Monte Carlo performance metrics

Let \(e_r=\hat\theta_r-\theta_r\) over successful replications.

- Bias: \(\bar e\)
- Absolute bias: \(|\bar e|\)
- Relative bias: \(\bar e/\bar\theta\), unavailable when the mean truth is effectively zero
- MSE: \(R^{-1}\sum e_r^2\)
- RMSE: \(\sqrt{MSE}\)
- Variance: sample variance of the estimates with denominator \(R-1\)
- Empirical SD: square root of that variance
- Mean estimated SE: average estimator-reported SE over finite SEs
- SE calibration ratio: mean estimated SE / empirical SD
- Coverage: fraction of valid intervals containing the replication-specific truth
- Average CI width: mean upper-minus-lower interval width
- Rejection rate: fraction of valid intervals excluding the null
- Type-I error: rejection rate when the truth is zero
- Power: rejection rate when the truth is nonzero
- Failure rate: fraction of attempted estimator fits without a finite estimate
- Convergence rate: fraction reporting convergence

Metrics that are not meaningful are returned as unavailable (`NaN`) rather than filled with invented values.

## 9. Monte Carlo standard errors

The finite number of Monte Carlo replications causes uncertainty in performance estimates.

### Bias

$
MCSE(\widehat{Bias})=SD(e_r)/\sqrt R.
$

### Coverage / rejection proportions

For proportion \(\hat p\),

$
MCSE(\hat p)=\sqrt{\hat p(1-\hat p)/R}.
$

### MSE and RMSE

$
MCSE(\widehat{MSE})=SD(e_r^2)/\sqrt R.
$

A delta-method approximation gives

$
MCSE(\widehat{RMSE})\approx
\frac{MCSE(\widehat{MSE})}{2\widehat{RMSE}}.
$

### Empirical variance and SD

Rather than assuming a Normal Monte Carlo sampling distribution, the implementation estimates uncertainty from the empirical influence function of the variance. Let \(\bar{\theta}\) be the mean estimate and let \(v\) denote the empirical variance. The per-replication variance influence contribution is approximated by the centered squared deviation

$
IF_{v,r} \propto (\hat\theta_r-\bar{\theta})^2-\frac1R\sum_s(\hat\theta_s-\bar{\theta})^2,
$

with the usual finite-sample scaling for the reported \(R-1\) variance. Its sample SD divided by \(\sqrt R\) gives the variance MCSE. The SD MCSE follows by the delta method,

$
IF_{SD,r} \approx IF_{v,r}/(2\,SD).
$

This avoids imposing a chi-square/Normal-theory approximation on deliberately heavy-tailed stress studies, although any variance-MCSE calculation still requires adequate finite higher moments in the simulated design.

Mean SE and mean CI width use the ordinary MCSE of a sample mean. The SE-calibration-ratio MCSE preserves the replication pairing between the reported SE and estimate and applies a first-order influence-function calculation.

## 10. Overlap and balance diagnostics

A diagnostic replication can be exactly regenerated from its replication seed. The app calculates:

- estimated propensity distributions by treatment group;
- observed-treatment inverse-probability weights;
- maximum weight;
- fraction of weights above 10 (a descriptive threshold);
- effective sample size

  $
  ESS=(\sum_i w_i)^2/\sum_iw_i^2;
  $

- standardized mean differences before and after weighting.

The ±0.10 SMD reference lines in the plot are descriptive conventions, not hypothesis-test cutoffs.

## 11. Stress-test interpretation

The seven-level Method Stress Test is deliberately cumulative. It moves from clean randomized Gaussian data to higher noise, confounding, nonlinearity, heteroskedasticity, poor overlap, and finally heavy tails. A robustness profile is therefore a statement about the chosen sequence of DGPs, not a universal estimator ranking.
