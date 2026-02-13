# Risk Estimation Architecture

This document describes the runtime architecture of the risk subsystem used by
`DataAgent` and `HumanAgentSteeringWheel`.

## Estimator Modes

Two estimators are supported behind a common API:

- `BoundaryRiskEstimator`: polar, physics-based directional risk.
- `SOTIFRiskEstimator`: grid-based `P(x,y) * C(x,y)` with polar-compatible
  summary output for radar-style rendering.

Runtime selection is controlled by environment variables:

- `USE_BOUNDARY_RISK=True|False`: master switch for risk estimation.
- `RISK_ESTIMATOR_TYPE=sotif|boundary`: estimator backend when risk is enabled.

In both agents, estimator calls use the same interface:

- `calculate_boundary_risk(ego_speed, bounding_boxes, timestamp)`
- returns `(risk_field, max_risk, threat_info)`

## DataAgent Pipeline (`team_code/data_agent.py`)

`DataAgent` extends `AutoPilot` and computes GT-based risk directly from CARLA
actors.

At each `run_step()` cycle, the risk pipeline is:

1. Collect ground-truth actor boxes from world state.
2. Convert actor state into estimator input format.
3. Call `calculate_boundary_risk(...)`.
4. Cache:
   - `last_boundary_risk_field`
   - `last_boundary_risk_info`

Visualization/debug behavior:

- Boundary mode: polar radar HUD.
- SOTIF mode: heatmap HUD using blended `C + R` display with orientation labels
  (`L`, `R`, `F`).
- Optional debug overlays and warning logic are available via flags.
- Optional live plotting via `RiskCurvePlotter`.

## SOTIF Estimator Design (`team_code/sotif_risk_estimator.py`)

`SOTIFRiskEstimator` implements a SOTIF-style decomposition:

- `P(x,y)`: ego trajectory-biased probability field.
- `C(x,y)`: directional kinetic risk based on approach component of relative
  velocity.
- `R(x,y) = P(x,y) * C(x,y)`.

Primary outputs:

- `risk_field`: polar summary for HUD compatibility.
- `max_risk`: scalar risk for thresholding and warning logic.
- `threat_info`: metadata dictionary that includes:
  - `heatmap` (`R`)
  - `heatmap_P`
  - `heatmap_C`
  - threat/risk labels and context fields.

Recent tuning behavior represented in code:

- Improved crossing-vehicle detectability.
- Expanded lateral sensitivity/presence handling for side-conflict scenarios.
- Receding/parallel suppression through directional approach terms.

## Human Agent Integration (`team_code/human_agent_keyboard.py`)

`HumanAgentSteeringWheel` uses the same estimator lifecycle and API contract as
`DataAgent`, including runtime estimator switching and shared HUD branches.

UI/HUD integration:

- Center camera full-screen.
- Left/right mirrors as small top-corner overlays.
- HUD region at bottom-right.
- Boundary vs SOTIF HUD switching matches `DataAgent`.
- Optional real-time `RiskCurvePlotter` integration.

Post-simulation analytics (saved at teardown):

- Risk value timeline.
- Risk level/type timeline and distribution.
- Risk direction timeline.
- Primary threat object analytics.
- Related-risk object analytics.
- JSON summary output.

## Real-Time Plotting Utility (`team_code/risk_curve_plotter.py`)

`RiskCurvePlotter` is a standalone real-time visualization helper used by both
agents when enabled.

- Live risk plotting in `polar`, `cartesian`, or `both` modes.
- Plot updates are throttled by configurable update rate.
- Designed to be low overhead and optional at runtime.

## Test Coverage (`team_code/test_risk_estimators.py`)

`test_risk_estimators.py` includes sanity checks for both estimators, including:

- Empty scene behavior.
- Lead-vehicle risk.
- Crossing pedestrian scenario.
- Receding suppression.
- Head-on conflict.
- Coordinate consistency checks.
- Multi-object cases.
- Output consistency and compatibility checks.

