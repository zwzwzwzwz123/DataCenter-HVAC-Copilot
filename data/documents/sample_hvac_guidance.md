# Sample HVAC Guidance

Data center cooling systems should manage airflow, supply temperature, and internal heat load together. Hot spots can appear when airflow is poorly balanced or when cooling response lags behind load changes.

Energy analysis should separate available HVAC power signals from optional derived metrics. PUE-like values require an explicit calculation method and should not be invented from BEAR trajectories without a documented mapping.

Control recommendations should come from policy tools or offline replay results. The Agent should explain the evidence and should not directly train a control model or write actions back into the environment.

