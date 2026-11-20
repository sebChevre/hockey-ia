# Phase B simulation conventions

- The generator uses a local `numpy.random.Generator(numpy.random.PCG64(seed))`.
  Each match receives a deterministic child seed, so its random stream is
  independent of random draws made later in other matches.
- Teams have stable IDs. Each roster has 12 forwards, 6 defensemen, and 2
  goalies; the configured `players_per_team=18` counts skaters, not goalies.
- Every skater has an estimated 1,000 seconds of ice time per team game
  (18 skaters share five on-ice positions); goalies alternate starts and the
  selected goalie is credited with 3,600 seconds. These are exposure estimates,
  not line changes or observed playing time.
- Shot coordinates use the common offensive frame: the shooting team attacks
  toward `(60, 15)` for every event. The rink dimensions are 60 m by 30 m.
- The baseline total shot-arrival rate is 62.5 attempts per match at
  `offensive_intensity=50`; the intensity multiplier is
  `0.5 + offensive_intensity / 100`. Each side's rate factor is the product of
  `(0.75 + 0.5 * attack_rating)`, `(1.15 - 0.3 * opponent_defense_rating)`,
  the home factor `(1 + home_advantage)` when applicable, and a manpower
  factor of `1.35` on a power play, `0.75` while short-handed, or `1.0` at
  equal strength. The total rate scales with the mean of both side factors,
  while the shooter is sampled in proportion to the side-specific rates.
  Exponential inter-arrival times yield Poisson shot events; variation means
  individual match totals can fall outside 55–70.
- For the baseline profile (`offensive_intensity=50`, `team_imbalance=30`,
  `home_advantage=0.05`), the mean total goals calibration range is 4–8 per
  match. It is an illustrative plausibility range, not a measured league
  statistic. The validator reports mean shots, goals, and penalties by match;
  penalties are compared to their configured Poisson expectation using a
  three-standard-error tolerance.
- `team_imbalance` controls the standard deviation of team ratings, increasing
  it from `0.025` at zero to `0.205` at 100; ratings are clipped to `[0.05, 0.95]`.
- Each goal is assigned zero, one, or two assists with probabilities 15%, 35%,
  and 50%, respectively. Assist recipients are distinct skaters on the scoring
  team, excluding the scorer. These proportions are synthetic assumptions,
  not measured hockey statistics.
- Penalty counts are Poisson with the configured expectation. Each assessed
 penalty is minor (`MINEURE`, 120 seconds) with 95% probability or major
 (`MAJEURE`, 300 seconds) with 5% probability. Each team serves at most two
 penalties at once; additional penalties wait in assessment order, and their
 clocks begin only when service starts. This keeps at least three skaters on
 the ice. An opposing power-play goal ends at most one active minor penalty
 (the one with the least remaining time); major penalties continue. Expirations
 or queued service extending beyond regulation are left unfinished in the log.
- A non-goal shot is saved with probability 0.70, otherwise it is missed.
  Goal-probability coefficients and these mechanisms are illustrative
  simulation assumptions, not empirically fitted hockey estimates. The
  configured default coefficients are `(-1.2, -0.08, -0.015, 1.2, 1.2, 0.25)`
  for `(beta_0, beta_distance, beta_angle, beta_skill, beta_goalie,
  beta_powerplay)`; probabilities are clipped to `[0.005, 0.70]`.
- A faceoff is emitted at the start of each period and immediately after each
  goal; this is not a model of every real-world faceoff. One forward from each
  team is selected uniformly. Each player's synthetic `faceoff_skill` is drawn
  independently from a clipped normal distribution with mean 0.5 and standard
  deviation 0.14. The home player's win probability is
  `0.5 + 0.5 * (home_faceoff_skill - away_faceoff_skill)`; the losing player is
  recorded on the same event. This illustrative rule is not fitted to real
  faceoff data.
- `shot_ground_truth` contains one row per shot and is separate from `shots`.
  No Streamlit, export, anomaly injection, or ML code is part of this phase.
