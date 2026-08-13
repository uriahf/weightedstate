# Estimation framework

The Aalen-Johansen estimator is a non-parametric estimator of transition probabilities in a multi-state model. It is expressed as a product integral of cumulative hazard increments, estimated using a multi-state analogue of the Nelson-Aalen estimator.

In the weighted estimator, risk-set and event counts are replaced by sums of the corresponding observation weights. At each event time, these weighted counts determine the cause-specific hazard increments and therefore the estimated state occupation probabilities.


# Kaplan-Meier as a special case

The Kaplan-Meier estimator is the two-state special case of the Aalen-Johansen estimator: an initial state and one absorbing event state. This relationship is preserved after weighting, so the weighted Kaplan-Meier estimator is the two-state special case of the weighted Aalen-Johansen estimator.
