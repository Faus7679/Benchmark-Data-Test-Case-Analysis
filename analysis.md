# Benchmark Data Test-Case Analysis
## Assignment 5 — Data Analysis Approaches and Stochastic Traffic Modeling

---

## Part 1: Analysis Approaches — Objectivity, Subjectivity, and Statistical Significance

### 1. Addressing Objectivity and Subjectivity in Data Analysis

When analyzing sensor-collected highway traffic data—as produced in the
Assignment 4 Benchmark-Research-Design experiment—the risk of researcher bias
influencing conclusions is substantial.  Two complementary approaches mitigate
this risk.

**Approach A: Confirmatory Data Analysis (CDA) with Pre-Registration.**
CDA requires the analyst to specify hypotheses, chosen statistics, and
acceptance criteria *before* examining the data (Lakens, 2022).  In the traffic
context this means declaring, prior to collection, that vehicle count will be
modeled as a Poisson process with a time-varying rate and that the null
hypothesis of no rush-hour peak will be rejected only at α = .05.  Pre-
registration removes the temptation to select metrics post-hoc that happen to
support a desired finding, thereby strengthening objectivity (Nosek et al.,
2018).  It also forces explicit operational definitions—for example, how a
"vehicle event" is demarcated by the sensor—reducing subjective coding
decisions.

**Approach B: Blind Analysis / Researcher Triangulation.**
Blind analysis, borrowed from experimental physics, hides key outcome values
from the analyst during the modeling stage (MacCoun & Perlmutter, 2015).  For
traffic data a second independent analyst applies the same pipeline to the same
raw sensor stream; only after both analyses are complete are conclusions
compared.  Disagreements are resolved through a pre-agreed arbitration rule
(e.g., majority rule among three analysts), rather than by the most senior
researcher's intuition.  This multi-analyst triangulation controls for both
conscious and unconscious subjectivity and is especially relevant when motion-
sensor data require manual quality-control decisions about anomalous readings.

**Rationale:** CDA addresses the *a priori* source of subjectivity (hypothesis
fishing), while blind triangulation addresses the *post-hoc* source (selective
reporting).  Together they span the lifecycle of the analysis and reinforce one
another (Lakens, 2022; Nosek et al., 2018).

---

### 2. Determining Statistical Significance and Accurate Outcome Measurements

**Approach A: Null Hypothesis Significance Testing (NHST) with Effect-Size
Reporting.**
NHST remains the dominant framework in traffic engineering and behavioral
sciences (Field, 2024).  For the highway experiment one would test whether mean
vehicle counts during peak hours differ significantly from off-peak hours using
a two-sample *t*-test or a Poisson regression, setting α = .05.  Critically,
the *p*-value alone is insufficient; effect sizes such as Cohen's *d* or
Incidence Rate Ratios must accompany every inferential result to convey
practical, not merely statistical, significance (Field, 2024).  Confidence
intervals (95% CI) around each estimate provide the uncertainty bound that
supports accurate outcome interpretation.

**Approach B: Bootstrap Resampling.**
Because traffic time-series data are non-independent (adjacent minutes are
autocorrelated) and the underlying count distribution is discrete, parametric
assumptions are frequently violated.  Bootstrap resampling generates an
empirical sampling distribution by drawing thousands of resamples from the
observed data, computing the statistic of interest each time, and reading
significance and CIs directly from the resulting distribution without Gaussian
assumptions (Efron & Hastie, 2021).  For the sensor data this preserves the
temporal autocorrelation structure when block-bootstrap (rather than i.i.d.)
resampling is used.

**Rationale:** NHST with effect sizes satisfies conventional reporting standards
and facilitates cross-study comparison, while block-bootstrap provides a robust
alternative when distributional assumptions—central to accurate measurement—
cannot be verified (Efron & Hastie, 2021; Field, 2024).

---

## Part 2: Stochastic Model for Simulated Highway Traffic Data

### 1. Model Description

The simulation represents one 24-hour day of single-lane highway traffic
captured by in-ground inductive loop sensors at one-minute resolution
(1,440 time steps).

#### Data Source 1 — Vehicle Counts: Poisson Distribution

Vehicle arrivals at a fixed sensor location satisfy the memoryless (Markov)
property and occur rarely within any short interval, making the **Poisson
distribution** the canonical and theoretically justified choice (Mannering et
al., 2020).  The intensity parameter λ(t) is not constant; it follows a
deterministic diurnal envelope composed of two Gaussian peaks (morning rush at
08:00, μ = 8 h, σ = 1 h; evening rush at 17:00, μ = 17 h, σ = 1.2 h)
superimposed on a low base rate of 3 vehicles/minute:

> λ(t) = 3 + 18·exp(−½((t − 8)/1)²) + 15·exp(−½((t − 17)/1.2)²)

At each minute *t*, the observed count is drawn as:

> N(t) ~ Poisson(λ(t))

#### Data Source 2 — Vehicle Speeds: Gaussian (Normal) Distribution

Under free-flow conditions on an uncongested highway, individual vehicle speeds
cluster around a posted speed limit with symmetric deviation attributable to
driver heterogeneity and road conditions.  The **Normal distribution** is
appropriate here (May, 1990).  The time-varying mean μ_v(t) decreases with
congestion (high N(t) relative to peak λ) and sensor measurement error is
modeled as additive white noise:

> V(t) ~ Normal(μ_v(t), σ_v(t)) + ε,  ε ~ Normal(0, 0.5²)

where μ_v(t) = 65·(1 − 0.45·c(t)) mph, σ_v(t) = 8·(1 − 0.30·c(t)) + 2 mph,
and c(t) = N(t)/max(λ) is the normalized congestion index.

#### Data Source 3 — Vehicle Headways: Log-Normal Distribution

The time gap (headway) between consecutive vehicles is strictly positive and
right-skewed: most gaps are short during peak hours, but a long tail represents
platoon-free travel.  **Log-normal** is the industry-standard distribution for
headways (May, 1990; Mannering et al., 2020):

> H(t) ~ Log-Normal(μ_ln(t), σ_ln = 0.50)

where μ_ln(t) = ln(max(60/N(t), 1)) links headway to per-minute vehicle count.

---

### 2. Implementation and Visualization

The model is implemented in **Python 3** using NumPy for stochastic sampling
and Matplotlib for visualization.  The complete source code is in
[`traffic_model.py`](traffic_model.py).

Running the script (`python traffic_model.py`) produces two output files:

| File | Contents |
|---|---|
| `traffic_time_series.png` | Four-panel 24-hour time series: Poisson counts with λ(t) overlay, Gaussian speeds with sensor noise, log-normal headways, and cumulative vehicle count |
| `traffic_distributions.png` | Marginal histograms of the three primary variables across the full day |

**Key model outputs (seed = 42):**
- Total simulated vehicles: **9,607**
- Peak rate: **35 vehicles/min** at 08:01
- Mean speed: **55.9 mph** (SD = 12.3)
- Median headway: **13.0 s** (IQR = 20.6 s)

The time series visually confirms expected diurnal patterns: dual rush-hour
spikes in counts, corresponding speed depressions, and compressed headways
during peak periods.

---

## References

Efron, B., & Hastie, T. (2021). *Computer age statistical inference: Algorithms,
evidence, and data science* (Student ed.). Cambridge University Press.
https://doi.org/10.1017/9781108914062

Field, A. (2024). *Discovering statistics using IBM SPSS statistics* (6th ed.).
SAGE Publications.

Lakens, D. (2022). Improving your statistical inferences. *Advances in Methods
and Practices in Psychological Science*, *5*(4), 1–21.
https://doi.org/10.1177/25152459221109861

MacCoun, R., & Perlmutter, S. (2015). Blind analysis: Hide results to seek the
truth. *Nature*, *526*, 187–189. https://doi.org/10.1038/526187a

Mannering, F. L., Washburn, S. S., & Scherer, W. T. (2020). *Principles of
highway engineering and traffic analysis* (7th ed.). Wiley.

May, A. D. (1990). *Fundamentals of traffic flow* (2nd ed.). Prentice-Hall.

Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The
preregistration revolution. *Proceedings of the National Academy of Sciences*,
*115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114
