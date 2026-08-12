# Get started

<img src="assets/package-mark.png" class="homepage-logo img-fluid" alt="weightedstate logo" />

`weightedstate` implements a weighted Aalen-Johansen estimator for time-to-event data. In the single-event setting, this estimator reduces to a weighted Kaplan-Meier estimator.

The package provides one core function, [weighted_aalen_johansen](reference/weighted_aalen_johansen.html#weightedstate.weighted_aalen_johansen), for single-event survival and competing-risk settings.


# Installation

With [uv](https://docs.astral.sh/uv/):

``` bash
uv add weightedstate
```

Alternatively, with pip:

``` bash
pip install weightedstate
```


# Single-event example


``` python
import polars as pl
from weightedstate import weighted_aalen_johansen

times = pl.Series([1, 2, 3, 4, 5])
reals = pl.Series([1, 0, 1, 1, 0])
weights = pl.Series([0.5, 1.5, 0.8, 1.2, 0.9])

weighted_aalen_johansen(times, reals, weights)
```


shape: (5, 15)

| times | count_0 | count_1 | count_2 | events_at_times | at_risk | csh_1 | csh_2 | conditional_survival | overall_survival | previous_overall_survival | transition_prob_1 | transition_prob_2 | cif_1 | cif_2 |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| i64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 |
| 1 | 0.0 | 0.5 | 0.0 | 0.5 | 4.9 | 0.102041 | 0.0 | 0.897959 | 0.897959 | 1.0 | 0.102041 | 0.0 | 0.102041 | 0.0 |
| 2 | 1.5 | 0.0 | 0.0 | 1.5 | 4.4 | 0.0 | 0.0 | 1.0 | 0.897959 | 0.897959 | 0.0 | 0.0 | 0.102041 | 0.0 |
| 3 | 0.0 | 0.8 | 0.0 | 0.8 | 2.9 | 0.275862 | 0.0 | 0.724138 | 0.650246 | 0.897959 | 0.247713 | 0.0 | 0.349754 | 0.0 |
| 4 | 0.0 | 1.2 | 0.0 | 1.2 | 2.1 | 0.571429 | 0.0 | 0.428571 | 0.278677 | 0.650246 | 0.371569 | 0.0 | 0.721323 | 0.0 |
| 5 | 0.9 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.0 | 1.0 | 0.278677 | 0.278677 | 0.0 | 0.0 | 0.721323 | 0.0 |


# Competing-risk example


``` python
times = pl.Series([1, 2, 3, 4, 5])
reals = pl.Series([1, 0, 1, 2, 0])
weights = pl.Series([0.5, 1.5, 0.8, 1.2, 0.9])

weighted_aalen_johansen(times, reals, weights)
```


shape: (5, 15)

| times | count_0 | count_1 | count_2 | events_at_times | at_risk | csh_1 | csh_2 | conditional_survival | overall_survival | previous_overall_survival | transition_prob_1 | transition_prob_2 | cif_1 | cif_2 |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| i64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 |
| 1 | 0.0 | 0.5 | 0.0 | 0.5 | 4.9 | 0.102041 | 0.0 | 0.897959 | 0.897959 | 1.0 | 0.102041 | 0.0 | 0.102041 | 0.0 |
| 2 | 1.5 | 0.0 | 0.0 | 1.5 | 4.4 | 0.0 | 0.0 | 1.0 | 0.897959 | 0.897959 | 0.0 | 0.0 | 0.102041 | 0.0 |
| 3 | 0.0 | 0.8 | 0.0 | 0.8 | 2.9 | 0.275862 | 0.0 | 0.724138 | 0.650246 | 0.897959 | 0.247713 | 0.0 | 0.349754 | 0.0 |
| 4 | 0.0 | 0.0 | 1.2 | 1.2 | 2.1 | 0.0 | 0.571429 | 0.428571 | 0.278677 | 0.650246 | 0.0 | 0.371569 | 0.349754 | 0.371569 |
| 5 | 0.9 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.0 | 1.0 | 0.278677 | 0.278677 | 0.0 | 0.0 | 0.349754 | 0.371569 |


### Links

[View on PyPI![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxZW0iIGhlaWdodD0iMWVtIiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIHN0eWxlPSJ2ZXJ0aWNhbC1hbGlnbjogLTAuMDVlbTsgbWFyZ2luLWxlZnQ6IDBlbTsgbWFyZ2luLXRvcDogMC4xZW07IiB2aWV3Ym94PSIwIDAgMjQgMjQiPjxwYXRoIGQ9Ik03IDdoMTB2MTAiIC8+PHBhdGggZD0iTTcgMTcgMTcgNyIgLz48L3N2Zz4=)](https://pypi.org/project/weightedstate/)\


### AI / Agents

[Skills<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iZ2Qtc3BhcmtsZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMC44NWVtIiBoZWlnaHQ9IjAuODVlbSIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIHN0eWxlPSJ2ZXJ0aWNhbC1hbGlnbjogLTAuMWVtOyBtYXJnaW4tbGVmdDogMC4yNWVtOyI+PHBhdGggZD0iTTkuOTM3IDE1LjVBMiAyIDAgMCAwIDguNSAxNC4wNjNsLTYuMTM1LTEuNTgyYS41LjUgMCAwIDEgMC0uOTYyTDguNSA5LjkzNkEyIDIgMCAwIDAgOS45MzcgOC41bDEuNTgyLTYuMTM1YS41LjUgMCAwIDEgLjk2MyAwTDE0LjA2MyA4LjVBMiAyIDAgMCAwIDE1LjUgOS45MzdsNi4xMzUgMS41ODJhLjUuNSAwIDAgMSAwIC45NjNMMTUuNSAxNC4wNjNhMiAyIDAgMCAwLTEuNDM3IDEuNDM3bC0xLjU4MiA2LjEzNWEuNS41IDAgMCAxLS45NjMgMHoiIC8+PHBhdGggZD0iTTIwIDN2NCIgLz48cGF0aCBkPSJNMjIgNWgtNCIgLz48L3N2Zz4=" class="gd-sparkle" />](skills.md)\
[llms.txt](llms.txt)\
[llms-full.txt](llms-full.txt)\


### Developers


**Uriah Finkel**

<span style="margin-top: -0.15em; display: block;">[](mailto:ufinkel@gmail.com "Email")</span>


### Meta

**Requires:** Python `>=3.9`\
[Package Info](package-info.md)
