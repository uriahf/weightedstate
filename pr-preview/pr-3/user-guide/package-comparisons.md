# Package comparisons

`weightedstate` is a deliberately small, Polars-native implementation of the weighted Aalen-Johansen estimator. Both lifelines in Python and survival in R cover a broader survival-analysis workflow.


# At a glance

|  | `weightedstate` | lifelines | R `survival` |
|----|----|----|----|
| Data interface | Polars Series/DataFrame | pandas/array-like | R vectors and formulas |
| Case weights | Yes | Yes | Yes |
| Competing risks | Yes | Yes | Yes |
| Output emphasis | All state probabilities together | CIF for one selected event per fit | Full survival or multi-state fit |
| Tied event times | Handled directly | Automatically jittered | Handled directly |
| Variance / confidence intervals | No | Yes | Yes |
| Left truncation | No | Yes, via `entry` | Yes |
| Natural fit | Focused Polars estimation | General Python survival analysis | Full-featured R survival analysis |

The table describes the documented public interfaces, not a ranking. The broader packages provide inference and modeling features that are intentionally outside `weightedstate`'s current scope.


# The same weighted competing-risk data

Each tab estimates cumulative state probabilities from the same observations and case weights.


- <a href="" id="tabset-1-1-tab" class="nav-link active" data-bs-toggle="tab" data-bs-target="#tabset-1-1" role="tab" aria-controls="tabset-1-1" aria-selected="true">weightedstate (Python 🐍)</a>
- <a href="" id="tabset-1-2-tab" class="nav-link" data-bs-toggle="tab" data-bs-target="#tabset-1-2" role="tab" aria-controls="tabset-1-2" aria-selected="false">lifelines (Python 🐍)</a>
- <a href="" id="tabset-1-3-tab" class="nav-link" data-bs-toggle="tab" data-bs-target="#tabset-1-3" role="tab" aria-controls="tabset-1-3" aria-selected="false">survival (R 🔵)</a>


``` python
import polars as pl
from weightedstate import weighted_aalen_johansen

times = pl.Series([1, 2, 3, 4, 5])
events = pl.Series([1, 0, 1, 2, 0])
weights = pl.Series([0.5, 1.5, 0.8, 1.2, 0.9])

weighted_aalen_johansen(
    times,
    events,
    weights,
)
```

**Output**

| Time | Overall survival |    CIF 1 |    CIF 2 |
|-----:|-----------------:|---------:|---------:|
|    1 |         0.897959 | 0.102041 | 0.000000 |
|    2 |         0.897959 | 0.102041 | 0.000000 |
|    3 |         0.650246 | 0.349754 | 0.000000 |
|    4 |         0.278677 | 0.349754 | 0.371569 |
|    5 |         0.278677 | 0.349754 | 0.371569 |


``` python
import pandas as pd
from lifelines import AalenJohansenFitter

data = pd.DataFrame(
    {
        "time": [1, 2, 3, 4, 5],
        "event": [1, 0, 1, 2, 0],
        "weight": [0.5, 1.5, 0.8, 1.2, 0.9],
    }
)

event_1 = AalenJohansenFitter()
event_1.fit(
    data["time"],
    data["event"],
    event_of_interest=1,
    weights=data["weight"],
)

event_2 = AalenJohansenFitter()
event_2.fit(
    data["time"],
    data["event"],
    event_of_interest=2,
    weights=data["weight"],
)
```

**Output**

| Time | CIF for event 1 | CIF for event 2 |
|-----:|----------------:|----------------:|
|    1 |        0.102041 |        0.000000 |
|    2 |        0.102041 |        0.000000 |
|    3 |        0.349754 |        0.000000 |
|    4 |        0.349754 |        0.371569 |
|    5 |        0.349754 |        0.371569 |

The initial-state probability is `1 - CIF 1 - CIF 2`. lifelines fits one event of interest at a time and also provides variance estimates and confidence intervals.


``` r
library(survival)

data <- data.frame(
  time = c(1, 2, 3, 4, 5),
  event = factor(
    c(1, 0, 1, 2, 0),
    levels = c(0, 1, 2),
    labels = c("censor", "event", "competing")
  ),
  weight = c(0.5, 1.5, 0.8, 1.2, 0.9)
)

fit <- survfit(
  Surv(time, event) ~ 1,
  data = data,
  weights = weight
)

summary(fit)
```

**Output**

| Time | Initial state |  Event 1 |  Event 2 |
|-----:|--------------:|---------:|---------:|
|    1 |      0.897959 | 0.102041 | 0.000000 |
|    2 |      0.897959 | 0.102041 | 0.000000 |
|    3 |      0.650246 | 0.349754 | 0.000000 |
|    4 |      0.278677 | 0.349754 | 0.371569 |
|    5 |      0.278677 | 0.349754 | 0.371569 |

R's `survival` package returns the full multi-state fit and includes extensive support for inference, formulas, delayed entry, and more complex event histories.


# Which should you use?

Use `weightedstate` for a compact Python/Polars workflow when the weighted state probabilities themselves are the target. Use lifelines when you want a broader Python survival toolkit, built-in confidence intervals, plotting, or left truncation. Use R's `survival` package when you need its mature multi-state framework and inferential machinery.


# References

- [lifelines `AalenJohansenFitter` documentation](https://lifelines.readthedocs.io/en/latest/fitters/univariate/AalenJohansenFitter.html)
- [R `survival` documentation](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.formula.html)
