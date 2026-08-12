## weighted_aalen_johansen()


Calculate a weighted Aalen-Johansen estimate.


Usage

``` python
weighted_aalen_johansen(
    times,
    reals,
    weights,
)
```


This function computes the probability of an event occurring over time, with support for weights. It can handle single events (like in Kaplan-Meier), or competing events (like in CIF).


## Parameters


`times: pl.Series`  
Event or censoring times.

`reals: pl.Series`  
Event types: 0 for censoring, 1 for the event of interest, and 2 for the competing event.

`weights: pl.Series`  
Non-negative observation weights.


## Returns


`pl.DataFrame`  
Weighted risk-set counts, cause-specific hazard increments, overall survival, and cumulative incidence estimates at each observed time.
