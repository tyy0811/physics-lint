# physics-lint-validation: Implementation Plan v1.0

**Status:** Draft, pre-execution
**Owner:** Jane Yeung
**Budget:** 3 days wall-clock, ~$30 Modal A100 ceiling
**Goal:** Two third-party case studies validating physics-lint v1.0 on pretrained SOTA neural PDE surrogates, packaged as a public repo (`physics-lint-validation`) and citeable in BMW ProMotion application + future cover letters.

---

## 1. Strategic framing (read first, do not skip)

The case studies serve **two distinct audiences simultaneously**, which dictates every downstream choice:

- **Audience A (academic, Munich/Stuttgart SciML):** Gerdts, Günnemann, Fehr/Kneifl, TUM/UniBw committees. Recognises LagrangeBench, MeshGraphNets, FNO, PDEBench. Values methodological rigor, falsifiable claims, equivariance theory.
- **Audience B (industrial, BMW Passive Safety):** Jehle, BMW R&D, NVIDIA-collaboration-style engineers. Recognises NVIDIA PhysicsNeMo, Modulus, Nabian arXiv:2510.15201. Values production-stack relevance, deployment gates, CI integration.

The two-case design is **deliberately bipartisan** — one case per audience. Do not collapse this into a single case study to save time; the dual coverage is the differentiator.

**The headline result we are aiming for:**

> physics-lint correctly *passes* an E(3)-equivariant model (SEGNN), correctly *flags* an approximately-equivariant model (GNS), and runs without modification on the production NVIDIA PhysicsNeMo MeshGraphNet stack used in the BMW-adjacent crash-surrogate paper Nabian et al. arXiv:2510.15201.

This sentence is the writeup target. Every step below is in service of being able to write it honestly.

---

## 2. Pre-flight (Day 0, ≤2 hours)

### 2.1 Environment audit

- Confirm physics-lint v1.0 is `pip install`-able from the published PyPI release.
- Confirm SARIF output works on a toy PyTorch model locally (CPU is fine).
- Identify which rules require a forward-pass hook vs. which operate on cached rollout tensors. **This determines the JAX adapter strategy.**
- Document in `DECISIONS.md` Entry 01 the answer to: *"Can physics-lint v1.0 lint a model whose rollouts are pre-computed and exported as `.npz`, without ever importing the model object?"*
  - If **YES** → JAX/PyTorch interop is trivial (export rollouts as NumPy, lint offline). Proceed with plan as written.
  - If **NO** → need a thin PyTorch wrapper. Time cost +2–4 hours. Decide before Day 1.

### 2.2 Modal account checks

- Verify A100 quota and current spend.
- Set hard budget cap: $30. If we hit $25, freeze new training runs.
- Create `physics-lint-validation` Modal app skeleton with two entrypoints: `lagrangebench_infer.py` and `physicsnemo_infer.py`.

### 2.3 NGC API key

- Sign up for NVIDIA NGC (free, takes 5 minutes), generate API key, store in Modal secret.
- Test `ngc registry model download-version "nvidia/modulus/modulus_ns_meshgraphnet:v0.1"` on a sandbox to confirm the key works. **Do this first** — auth issues mid-Day-2 are a budget hit.

### 2.4 Repo skeleton

```
physics-lint-validation/
├── README.md                       # case study summaries + headline result + links
├── DECISIONS.md                    # methodological log (user's standard format)
├── pyproject.toml                  # PyTorch + JAX + nvidia-physicsnemo as extras
├── case-studies/
│   ├── 01-lagrangebench/
│   │   ├── README.md               # what was tested, what physics-lint found
│   │   ├── modal_app.py            # Modal entrypoint
│   │   ├── run_inference.py        # rollout generation
│   │   ├── lint_rollouts.py        # physics-lint invocation
│   │   ├── outputs/
│   │   │   ├── rollouts/           # .npz files (gitignored if >50MB)
│   │   │   ├── lint.sarif          # canonical lint output
│   │   │   └── figures/
│   │   └── tests/                  # pytest, ≥3 unit tests
│   └── 02-physicsnemo-mgn/
│       ├── README.md
│       ├── modal_app.py
│       ├── run_inference.py
│       ├── lint_rollouts.py
│       ├── outputs/
│       └── tests/
├── shared/
│   ├── jax_pytorch_bridge.py       # IF needed (Day 0 decides)
│   ├── lint_helpers.py             # rule invocation wrappers
│   └── plot_helpers.py
├── .github/workflows/ci.yml        # run lint on cached rollouts in CI
└── Dockerfile                      # reproducibility, optional
```

This mirrors the agent-bench / sim-to-data structure already established. No reinvention.

---

## 3. Case Study 01: LagrangeBench (Day 1, 6–8 hours)

### 3.1 Targets (in priority order)

Three datasets, two architectures = up to six rollout sets. **Aim for the top 3 first; do the rest only if time remains.**

| Priority | Dataset | Architecture | Headline rule |
|---|---|---|---|
| P0 | TGV 2D (Taylor-Green vortex) | SEGNN | PH-CON-003 (monotone energy decay) + PH-SYM equivariance baseline |
| P0 | TGV 2D | GNS | Same rules → expect equivariance flag |
| P1 | Dam break 2D | GNS | PH-CON-001 (mass), PH-BC (wall) |
| P2 | Reverse Poiseuille 2D | SEGNN | PH-BC (no-slip) |
| P3 | Dam break 2D | SEGNN | Cross-validate P1 result |
| P3 | TGV 3D | GNS | Stretch only |

P0 + P1 = the headline. P2/P3 only if Day 1 finishes early.

### 3.2 Step-by-step

**Step 1 — Install (30 min).** On Modal A100 image:

```bash
pip install --upgrade pip
pip install -U "jax[cuda12]" jaxlib  # match Modal CUDA version
git clone https://github.com/tumaer/lagrangebench && cd lagrangebench
pip install -e ".[dev]"
bash download_data.sh tgv2d datasets/
bash download_data.sh dam2d datasets/
```

Smoke test: `python main.py mode=infer dataset=tgv2d` on the included toy config to confirm JAX sees the GPU.

**Step 2 — Download checkpoints (15 min).** From the LagrangeBench README's gdown links:

```bash
gdown <gns_tgv2d_zip_url>
gdown <segnn_tgv2d_zip_url>
gdown <gns_dam2d_zip_url>
unzip -d checkpoints/ ./*.zip
```

**Step 3 — Generate rollouts (1–2 hours).** For each (dataset, model) pair:

```bash
python main.py mode=infer \
    load_ckp=checkpoints/segnn_tgv2d/best/ \
    config=configs/segnn_tgv2d.yaml \
    eval.n_rollout_steps=100 \
    eval.n_trajs=20
```

Export trajectories to `outputs/rollouts/segnn_tgv2d.npz` with the schema:

```python
{
    "positions":    np.ndarray,  # (T, N, D)  -- T=100, N≈3000, D=2
    "velocities":   np.ndarray,  # (T, N, D)
    "particle_type":np.ndarray,  # (N,)       -- fluid/wall flags
    "dt":           float,
    "domain_box":   np.ndarray,  # (2, D)
    "metadata":     dict,        # dataset name, ckpt hash, git SHA
}
```

**The schema is critical** — physics-lint reads from this, and downstream cross-comparison with PhysicsNeMo (Day 2) reads from this. Do not deviate.

**Step 4 — Run physics-lint (1 hour).** For each rollout `.npz`:

```python
from physics_lint import lint_rollout

result = lint_rollout(
    rollout_path="outputs/rollouts/segnn_tgv2d.npz",
    rules=[
        # Conservation
        "PH-CON-001",  # mass: particle count + ∫ρ
        "PH-CON-002",  # energy: kinetic energy trajectory
        "PH-CON-003",  # dissipation sign: KE monotone decreasing for TGV
        # Symmetry — applied as input perturbation tests
        "PH-SYM-001",  # C4 rotation
        "PH-SYM-002",  # reflection
        "PH-SYM-003",  # SO(2) Lie derivative
        "PH-SYM-004",  # translation
        # Boundary
        "PH-BC-001",   # wall non-penetration (dam break)
    ],
    output_format="sarif",
    output_path="outputs/lint_segnn_tgv2d.sarif",
)
```

**Step 5 — Equivariance test wiring (this is the tricky part, 2 hours).**

PH-SYM-* rules need a "rotated input → rotated output" comparison. With a frozen JAX checkpoint, this means:

1. Load the checkpoint into Haiku.
2. Generate one rollout from initial conditions $x_0$.
3. Apply rotation $R \in SO(2)$ to $x_0$, generate a second rollout.
4. Apply $R^{-1}$ to the second rollout's positions/velocities.
5. Compute $\epsilon_{\text{rot}} = \|R^{-1} f(R x_0) - f(x_0)\|$ as a function of rotation angle $\theta \in \{0, \pi/4, \pi/2, \pi, 3\pi/2\}$.
6. **Threshold:** $\epsilon_{\text{rot}} < 10^{-5}$ → PASS (machine precision; SEGNN expected). $10^{-5} < \epsilon_{\text{rot}} < 10^{-2}$ → APPROXIMATE (GNS expected). $\epsilon_{\text{rot}} > 10^{-2}$ → FAIL.

**Hold the threshold honest before running** — write the threshold into the test code, then run, then report. Do not retroactively tune the threshold to match the observed value. (This is the measurement-before-amendment discipline applied to the validation itself.)

**Step 6 — Capture + write up (1–2 hours).**

For each (dataset, model) pair, write a short subsection of `case-studies/01-lagrangebench/README.md`:

```markdown
### TGV 2D — SEGNN
- Checkpoint: segnn_tgv2d, best/, SHA-256 <hash>
- Rollout: 20 trajectories × 100 steps
- PH-CON-001: PASS (Δparticle_count = 0, Δ∫ρ < 1e-9)
- PH-CON-002: KE MSE = X (LagrangeBench reports Y; ratio Z)
- PH-CON-003: PASS (KE monotone decreasing across all 20 trajectories)
- PH-SYM-001 (C4): PASS (ε_rot = 4.3e-7, < 1e-5 threshold)
- PH-SYM-002 (reflection): PASS (ε_refl = 3.8e-7)
- PH-SYM-003 (SO(2)): PASS (Lie derivative norm < 1e-6)
- PH-SYM-004 (translation): PASS

### TGV 2D — GNS
- ...
- PH-SYM-001 (C4): FLAG (ε_rot = 1.2e-2, in approximate-equivariance band)
- ...
```

### 3.3 Day 1 acceptance criteria

Before moving to Day 2:

- [ ] At least one (dataset, model) pair has full SARIF output committed.
- [ ] SEGNN-vs-GNS equivariance comparison shows quantitatively distinct results (the headline).
- [ ] `outputs/rollouts/*.npz` is reproducible (re-running the Modal entrypoint gives bit-identical results given fixed seed).
- [ ] `DECISIONS.md` has 5+ entries (matches user's pattern).

**If by hour 6 of Day 1 the headline equivariance result is not visible** → defrag: drop GNS dam break and reverse Poiseuille, focus all remaining time on getting one clean SEGNN-vs-GNS TGV comparison. The headline matters more than coverage.

---

## 4. Case Study 02: NVIDIA PhysicsNeMo MeshGraphNet (Day 2, 4–6 hours)

### 4.1 Targets

Two checkpoints, both already on NGC:

| Priority | Checkpoint | Domain | Headline rule |
|---|---|---|---|
| P0 | `modulus_ns_meshgraphnet` (vortex shedding 2D) | Incompressible NS, cylinder wake | PH-NUM-002 (resolution) + PH-CON-001 (mass) |
| P1 | `modulus_ahmed_body_meshgraphnet` | Steady RANS, car-like geometry | PH-SYM-002 (y-reflection) + PH-BC (no-slip) |

P0 first — vortex shedding has cleaner physics for our rules and stronger PH-NUM-002 signal. P1 is the BMW-recognition signal (Ahmed Body = automotive).

### 4.2 Step-by-step

**Step 1 — Container + install (30 min).** On Modal A100:

```bash
# Use NVIDIA's container directly; it has DGL + PyTorch pre-installed
# nvcr.io/nvidia/modulus/modulus:25.08
pip install nvidia-physicsnemo
ngc config set  # use stored API key
```

**Step 2 — Download checkpoints (15 min).**

```bash
ngc registry model download-version "nvidia/modulus/modulus_ns_meshgraphnet:v0.1"
ngc registry model download-version "nvidia/modulus/modulus_ahmed_body_meshgraphnet:v0.1"
```

Verify hashes; commit hashes to `DECISIONS.md`.

**Step 3 — Run NGC inference scripts (1 hour).** Each NGC checkpoint ships with a sample input and an inference script. Run as documented:

```bash
cd modulus_ns_meshgraphnet_v0.1/
python inference.py --input sample_input/ --output outputs/
```

**Export rollouts to the same `.npz` schema as Day 1** so the lint helpers are reusable. For mesh-based MGN, the schema becomes:

```python
{
    "node_positions":  np.ndarray,  # (T, N_nodes, D)
    "node_velocities": np.ndarray,  # (T, N_nodes, D)  -- if present
    "node_pressures":  np.ndarray,  # (T, N_nodes)     -- vortex shedding
    "edge_index":      np.ndarray,  # (2, N_edges)
    "node_type":       np.ndarray,  # (N_nodes,)       -- INFLOW/WALL/INTERIOR/...
    "dt":              float,
    "metadata":        dict,
}
```

**Step 4 — Run physics-lint (1 hour).** Same `lint_rollout` API as Day 1. Rules to test:

- **PH-CON-001** (mass) on vortex shedding: divergence-free check on velocity field.
- **PH-CON-002** (energy) on vortex shedding: KE budget.
- **PH-NUM-002** (refinement): re-run inference at 0.5×, 1×, 2× mesh resolution if the checkpoint supports it (vortex shedding does, parameterized; Ahmed Body does not — skip there).
- **PH-SYM-002** (reflection) on Ahmed Body: y-reflect the geometry, run inference, compare. Threshold: $\epsilon_{\text{refl}} < 10^{-3}$ → APPROXIMATE PASS (mesh-induced asymmetry expected).
- **PH-BC-001** on both: no-slip wall enforcement, $|v|_{\text{wall}} < 10^{-4}$.

**Step 5 — Cross-comparison subsection (1 hour).** The interesting story across both case studies:

> physics-lint v1.0 ran without rule modification across two completely different stacks (JAX/Haiku Lagrangian particle dynamics; PyTorch/DGL Eulerian mesh-based GNN). The same SARIF schema, the same rule set, the same thresholds. This demonstrates the rule abstractions are domain-portable — a property required for a CI gate to be useful in industry.

This is the bridge sentence that makes the writeup more than a sum of its parts.

### 4.3 Day 2 acceptance criteria

- [ ] Both PhysicsNeMo checkpoints run inference end-to-end on Modal.
- [ ] SARIF outputs committed for vortex shedding (P0).
- [ ] PH-NUM-002 resolution sweep produces a meaningful curve (3 resolutions).
- [ ] Cross-stack consistency check passes: same rules give qualitatively-comparable results on both stacks.

**Drop-decision if behind schedule:** drop Ahmed Body (P1), keep vortex shedding (P0). The story still holds — the BMW-recognition signal is weaker but the technical claim is unaffected.

---

## 5. Day 3: writeup, optional retrain, application integration

### 5.1 Writeup (3–4 hours, mandatory)

Build `README.md` at repo root. Structure (mirrors user's existing repo READMEs):

```markdown
# physics-lint-validation

Two third-party case studies validating physics-lint v1.0 on pretrained
SOTA neural PDE surrogates.

## Headline result
[1 paragraph, the bridge sentence from §4.2 step 5]

## Case studies
### 01 — LagrangeBench (TUM, NeurIPS 2023)
[1 paragraph + table + link]
### 02 — NVIDIA PhysicsNeMo MeshGraphNet (NGC pretrained)
[1 paragraph + table + link]

## Cross-stack consistency
[Table: rule × case study × result]

## What physics-lint did NOT catch
[Be honest. List things the rules cannot detect. This is the methodological-honesty
signal that matters for Audience A.]

## Reproducibility
[Modal entrypoints, checkpoint hashes, git SHA, commit your conda lockfile]

## Citations
- Toshev et al. (LagrangeBench, NeurIPS 2023)
- Pfaff et al. (MeshGraphNets, ICLR 2021)
- Nabian et al. (arXiv:2510.15201, BMW/GM-style crash with PhysicsNeMo)
- Lahoz Navarro & Jehle et al. (Applied Sciences 2024) — frame physics-lint as
  the deterministic-violation gate complementing their BNN UQ approach
```

**The "what physics-lint did NOT catch" section is non-negotiable.** It is the same methodological-honesty move as sim-to-data Decision 19 and inverseops V3. Writing it well is what separates this work from a marketing demo. Examples to cover:
- Plasticity/irreversibility rules are not yet implemented (PH-CSH-* roadmap, separate issue).
- Contact-non-penetration on deforming meshes is not tested (no public checkpoint exists).
- Equivariance tests are statistical (over a finite set of rotation angles); they cannot prove equivariance, only fail to disprove it.
- LagrangeBench is fluid SPH, not solid impact — domain transfer is implied, not demonstrated.

### 5.2 Optional retrain (Day 3, only if buffer remains and budget allows)

If Days 1–2 finished in 10 hours total and Modal spend is below $15:

- Retrain MeshGraphNets on DeepMind `deforming_plate` for 5 epochs via NVIDIA's PhysicsNeMo PyTorch port.
- Estimated cost: 10–15 A100-hours, $15–25.
- The lint result here is **deliberately weak** — a partially-trained model violating physics. The headline becomes: *"physics-lint flagged PH-CON-002 violations on a 5-epoch checkpoint; flag count drops monotonically with training epochs (5/15/25), demonstrating the tool functions as a CI training-completeness gate."*

This is a strong optional addition but **not** worth blowing the budget for. Decision rule: only proceed if both (a) wall-clock buffer ≥6 hours remains, AND (b) Modal spend < $15.

### 5.3 Application integration (1 hour, mandatory)

Update three places:

1. **physics-lint repo README** — add a "Validated on" badge row linking to physics-lint-validation case studies.
2. **CV — physics-lint entry** — append: *"Validated against pretrained third-party SOTA surrogates (LagrangeBench GNS/SEGNN, NVIDIA PhysicsNeMo MeshGraphNet); see physics-lint-validation."*
3. **BMW cover letter (if not yet sent) — replace the Tier-2 smoke-project paragraph with:**

> *"To validate the linter against external models, I ran physics-lint v1.0 on pretrained checkpoints from LagrangeBench (Toshev et al., NeurIPS 2023; SEGNN and GNS on Taylor-Green vortex and dam break) and NVIDIA PhysicsNeMo MeshGraphNet (vortex shedding, Ahmed Body). The PH-SYM rule family correctly distinguished the E(3)-equivariant SEGNN from the approximately-equivariant GNS (ε_rot < 10⁻⁶ vs. ≈ 10⁻²), and the same rules ran without modification on the production PyTorch/DGL stack used in Nabian et al. arXiv:2510.15201 — the closest open analogue to BMW's MGN-on-LS-DYNA crash workflow. Extending physics-lint with nonlinear-structural-mechanics primitives (contact non-penetration, plastic-strain irreversibility, frame indifference) is the natural first six months of the proposed PhD."*

If the cover letter is already sent: hold this paragraph for the (likely) follow-up email to Jehle one week post-submission.

---

## 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| LagrangeBench JAX checkpoint loading non-trivial | Medium | Day 0 audit; allow +2h on Day 1; fallback is GNS-only on TGV |
| NGC API key auth fails mid-Day-2 | Low | Tested in Day 0 pre-flight |
| PhysicsNeMo NGC checkpoint format changed since docs | Low-Medium | Read NGC catalog page on Day 0 to confirm current version |
| Equivariance threshold of 10⁻⁵ too tight or too loose for SEGNN | Medium | Pre-compute on a 5-trajectory pilot before full run; if SEGNN gives 10⁻⁴ instead of 10⁻⁶, document honestly and adjust band — but log the decision in DECISIONS.md transparently, do not silently retune |
| Modal A100 quota / availability | Low | $30 hard cap; H100 fallback if A100 unavailable (slightly more expensive) |
| Rule false-pass on regular grids (the user's original concern about Transolver) | Mitigated by design | LagrangeBench is on irregular particle clouds; PhysicsNeMo MGN is on irregular meshes. Neither uses Cartesian grids. |
| Story risk: SEGNN doesn't actually pass equivariance to machine precision | Low (theory says it should) | If it doesn't, the headline becomes *"physics-lint detects implementation deviations from theoretical equivariance in SEGNN"* — still publishable, more interesting |
| Retraining `deforming_plate` blows budget | Medium | Day 3 retrain is OPTIONAL; gated on $15 spend remaining |
| Physics-lint v1.0 missing a rule needed for these benchmarks | Low | If PH-CON-002 doesn't compute KE the way LagrangeBench does, write a thin adapter; do NOT amend the rule mid-validation (measurement-before-amendment) |

---

## 7. Decision gates (when to deviate from this plan)

**End of Day 0:** if pre-flight audit reveals physics-lint cannot lint pre-computed `.npz` rollouts, add a Day 0.5 (4h) for a thin model-import adapter. Push everything by half a day.

**End of Day 1, hour 4:** if checkpoint loading is still broken, switch from SEGNN to GNS-only and accept that the equivariance headline is weaker. (GNS-only still gives a "physics-lint flags GNS as approximately equivariant" story.)

**End of Day 1, hour 8:** if no SARIF output committed, defer Case Study 02 by one day and treat this as a 4-day project. Do not start Day 2 with Day 1 incomplete.

**End of Day 2, hour 4:** if PhysicsNeMo NGC checkpoint is unusable (format mismatch, license issue, anything), switch to neuraloperator FNO on Darcy as the secondary case. Recognition value is lower, but story still holds. **Do not** silently skip Case Study 02 — the dual-audience strategy depends on having two cases.

**Day 3, hour 0:** if total Modal spend ≥ $20 → no retrain. Writeup only.

**Always:** if a result surprises us (e.g., SEGNN flagged, GNS passed) → **stop, audit the measurement before amending the narrative**. This is the user's standard debugging discipline applied here.

---

## 8. Out of scope (do not get tempted)

- Implementing PH-CSH-* (plasticity, contact, frame indifference) rules. That belongs in the **physics-lint roadmap issue** opened separately. This validation runs the existing v1.0 rules.
- Comparing against Transolver. No public checkpoint, retraining cost out of budget.
- ReGUNet reproduction. No public code.
- Crash-specific results. The honest framing is "validated on adjacent benchmarks (Lagrangian SPH, mesh-based NS)" with explicit acknowledgement that crash-specific contact + plasticity is out of public-checkpoint reach today — and that is precisely the gap the proposed PhD project would close.
- Custom dataset generation. Use shipped data only.
- Performance optimisation. Inference latency is not the story.

---

## 9. Cross-review checklist (before publishing)

User's standard 5-round Claude/ChatGPT cross-review applies here. Specifically check:

- [ ] Are the equivariance thresholds defensible? (Pre-registered, not retroactively tuned.)
- [ ] Is the "what physics-lint did NOT catch" section honest enough that a Stuttgart/Munich reviewer would believe it?
- [ ] Are all checkpoint hashes, git SHAs, and Modal lockfiles committed? Reproducibility audit.
- [ ] Does the README headline survive the one-sentence test: can you state the result honestly in 30 words?
- [ ] Cover-letter paragraph (§5.3) — does it overclaim? Specifically, the phrase "the closest open analogue to BMW's MGN-on-LS-DYNA crash workflow" — is it defensible? (Yes, but only if Nabian et al. arXiv:2510.15201 is the BMW-adjacent reference, which we have established.)
- [ ] DECISIONS.md ≥20 entries, matching the user's quality bar from agent-bench/sim-to-data.

---

## 10. Definition of done

This validation work is **done** when all of the following are true:

1. `physics-lint-validation` is a public repo on `github.com/tyy0811/`.
2. README headline result is one falsifiable sentence supported by SARIF artifacts.
3. Two case studies, each with reproducible Modal entrypoint, SARIF output, README, and ≥3 unit tests.
4. DECISIONS.md ≥20 entries.
5. CI workflow runs lint on cached `.npz` rollouts in <5 minutes (proof that physics-lint is genuinely a CI gate).
6. Cross-review by ChatGPT (≥3 rounds) on the methodology, not just the prose.
7. The cover-letter paragraph (§5.3) survives a "would Jehle find this overclaiming?" test.

Time budget on this definition: 3 days wall-clock + 1 day buffer = 4 days max. If we hit Day 5, something is wrong with the plan, not the execution.
