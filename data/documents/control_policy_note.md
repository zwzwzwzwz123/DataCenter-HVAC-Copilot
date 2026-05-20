# Control Policy Note

Source type: project internal note.
Published at: 2026.
Category: control_policy.

The Agent is not the control policy. It should route the task, gather evidence, call deterministic tools, and explain the result. Control recommendations should come from rule-based policies, MPC-like adapters, diffusion policy adapters, or offline replay results.

If DiffFNO or Guided-DiffFNO inference is not connected, the system must report that limitation and use a configured fallback instead of fabricating model outputs.
