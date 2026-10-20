# Sampling the past, proposing the future
## Research-grounded animation specification for the CD3 Numerics Forum

**Document version:** 1.0, 20 October 2026  
**Status:** Proposed design for review; no animation implementation or sampler benchmark has been completed.  
**Primary presentation:** 20 minutes, including a five-minute inference introduction. A 25-minute route is specified separately.  
**Audience:** Cosmologists, astrophysicists, simulation developers, and researchers who do not routinely implement samplers.

## 1. Design brief and decisions

The user wants interactive explanations of importance sampling, random-walk Metropolis–Hastings, Hamiltonian Monte Carlo, and nested sampling. The examples must connect to Bayesian inverse problems, compare mechanisms rather than software popularity, and expose opportunities for future numerical advances. The new requirement is to explain importance nested sampling (INS) and the neural-network regression underlying Nautilus, without turning the talk into an ML architecture survey.

**Decision:** Build one application with two related, scientifically interpretable inference examples and three principal demonstration modules. The third module has a classical-NS chapter and an INS/learned-bound chapter. Include a separate hierarchical example only as a backup. Do not create a different target distribution for each algorithm.

**Central narrative:** account for uncertain calibration; learn to weight independent proposals; compare diffusive and dynamical proposals; integrate over prior mass; then learn more efficient proposal regions while retaining likelihood-based inference.

**Recommended delivery assumption, not a previously specified user requirement:** an offline, 16:9 browser presentation, driven primarily by verified, recorded numerical traces. Include video fallbacks for embedding in slides. The browser is not required to train neural networks during the talk.

### 1.1 Approaches considered

| Approach | Advantage | Main drawback | Decision |
|---|---|---|---|
| One generic posterior for everything | Minimal context switching | Conflates posterior density, likelihood, and prior mass; weak motivation for a network on an ellipse | Reject as the only example |
| Separate realistic astrophysical analysis for each sampler | Strong application specificity | Too much new physics and notation for 20 minutes; comparisons become ambiguous | Keep real applications as brief references, not live problems |
| One measurement family, linear and nonlinear response variants | Same parameters and data; isolates geometry and multimodality; independent reference calculations possible | Requires saying explicitly that these are synthetic teaching models | Adopt |

### 1.2 Non-goals

No actual survey likelihood, N-body integration, GPU speed benchmark, neural posterior estimator, normalizing-flow tutorial, full SMC demonstration, or claim that one sampler is universally best. No implementation of production Nautilus from a six-frame cartoon. No claim that a two-dimensional demonstration establishes high-dimensional scaling.

## 2. Example-to-animation map

| Inference example | Intro / marginalization | Importance sampling | Random walk / HMC | Classical NS | INS and learned regions |
|---|---|---|---|---|---|
| A. Linear signal plus uncertain background | Main example | Main example | Main comparison | First explanation of prior mass | Sanity check, not main NN showcase |
| B1. Quadratic response, weak calibration | Reuse same data story | Optional proposal-mismatch replay | Optional geometry stress test | Show curved constrained regions | Main network-regression demonstration |
| B2. Same quadratic response, strong calibration | Explain two allowed signs | Optional missed-support example | Brief mode-trapping replay | Global exploration, with coverage caveats | Coverage and finite-sample failure test |
| C. Hierarchical population scale and latent signal | Backup hierarchy detail | Not required | Centered/non-centered comparison | Not required | Not required |

All methods in a comparison must use the same data, normalized likelihood, normalized prior, and target version. Changing a model or data control creates a new run; never pool draws across different targets.

## 3. Shared inference model

### 3.1 Physical interpretation and notation

All variables in the public presets are dimensionless, standardized quantities. Let `a` be a signed latent signal amplitude and `b` an additive background/calibration offset. Observe a signal channel `d_s` and an independent calibration channel `d_c`:

\[
 d_s = h(a)+b+\epsilon_s,\qquad d_c=b+\epsilon_c,
 \quad \epsilon_s\sim\mathcal N(0,\sigma_s^2),\quad
 \epsilon_c\sim\mathcal N(0,\sigma_c^2).
\]

Use independent, proper priors:

\[
 a\sim\mathcal N(0,\tau_a^2),\qquad b\sim\mathcal N(0,\tau_b^2).
\]

The fully normalized likelihood is

\[
 \mathcal L(a,b)=\frac{1}{2\pi\sigma_s\sigma_c}
 \exp\!\left[-\frac{(d_s-h(a)-b)^2}{2\sigma_s^2}
             -\frac{(d_c-b)^2}{2\sigma_c^2}\right].
\]

Write the prior as `pi`, posterior as `p`, and evidence as `Z` throughout. Do not use the same symbol for prior and posterior.

\[
 p(a,b\mid d)=\mathcal L(a,b)\pi(a,b)/Z,
 \qquad Z=\int\mathcal L(a,b)\pi(a,b)\,da\,db.
\]

**Interpretation constraint:** These are synthetic calibration/response models. A quadratic response resembles loss of sign information in an intensity measurement, but it is not a realistic galaxy-power-spectrum or weak-lensing likelihood. `a` is not sigma8 and is allowed to be negative. Do not relabel an arbitrary contour as a published cosmological posterior.

This two-parameter example illustrates nuisance marginalization, not a deep hierarchical population model. Use the general graph in the introduction and the separate Example C for genuinely hierarchical structure.

### 3.2 Fixed public presets

Shared values: `d=(1,0)`, `sigma_s=0.12`, `tau_a=2`, `tau_b=1`.

| ID | Response | Calibration noise | Purpose |
|---|---|---:|---|
| A-linear | `h(a)=a` | 0.60 | Strongly correlated but analytically tractable posterior |
| A-calibrated | `h(a)=a` | 0.16 | Show what improved calibration changes |
| B-curved | `h(a)=a^2` | 0.60 | Curved ridge near `b=1-a^2`; motivates non-ellipsoidal proposals |
| B-separated | `h(a)=a^2` | 0.16 | Two separated solutions near `a=+1` and `a=-1` |

The two B presets differ only in the calibration uncertainty. Their priors remain sign-symmetric, so the exact posterior probability of `a>0` is one half. This provides a stringent, interpretable mode-coverage diagnostic.

### 3.3 Independent reference solution

For Example A, define

\[
 D=\begin{pmatrix}1&1\\0&1\end{pmatrix},\quad
 P_0=\operatorname{diag}(\tau_a^2,\tau_b^2),\quad
 S=\operatorname{diag}(\sigma_s^2,\sigma_c^2).
\]

Then

\[
 C=(P_0^{-1}+D^T S^{-1}D)^{-1},\quad
 m=CD^T S^{-1}d,\quad
 Z=\mathcal N_2(d;0,S+DP_0D^T).
\]

For either response, integrate out `b` first. With `v=(1,1)^T` and `V=S+tau_b^2 vv^T`,

\[
 p(d\mid a)=\mathcal N_2(d;(h(a),0)^T,V),
 \qquad Z=\int \pi(a)p(d\mid a)\,da.
\]

This is a one-dimensional quadrature reference for Example B, independent of all samplers. Its conditional distribution is also analytic:

\[
 v_b=(\tau_b^{-2}+\sigma_s^{-2}+\sigma_c^{-2})^{-1},
\]
\[
 m_b(a)=v_b\big[(d_s-h(a))/\sigma_s^2+d_c/\sigma_c^2\big].
\]

Thus `b | a,d ~ Normal(m_b(a),v_b)`. Reference moments follow from one-dimensional integration and conditional Gaussian moments. Check quadrature with tighter tolerances and explicit interval splitting near both modes; a single unresolved two-dimensional plotting grid is not ground truth.

**Completed mathematical preflight, not sampler validation:** analytic and quadrature evidence for both linear presets agreed within `3e-15` relative tolerance. Values below were computed for the definitions above.

| Preset | log Z | E[a] | E[b] | Var[a] | Var[b] |
|---|---:|---:|---:|---:|---:|
| A-linear | -2.8353382971 | 0.9347747193 | 0.0618600917 | 0.2609011228 | 0.2483311522 |
| A-calibrated | -2.6723411801 | 0.9902556374 | 0.0061794424 | 0.0389774506 | 0.0248067534 |
| B-curved | -2.6656607331 | 0 | 0.1792548781 | 0.8109936565 | 0.2516058055 |
| B-separated | -2.6495258300 | 0 | 0.0165767854 | 0.9738600672 | 0.0253569685 |

For A-linear, correlation is approximately `-0.9721`. The marginal standard deviation of `a` is `0.5108`, compared with `0.1198` when conditioning on a fixed value of `b`. This makes the marginalization lesson visible without inventing an illustrative error bar.

### 3.4 Gradients and coordinates

Let `r_s=d_s-h(a)-b`, `r_c=d_c-b`. The physical-coordinate log-posterior gradient is

\[
 \partial_a\log p=h'(a)r_s/\sigma_s^2-a/\tau_a^2,
 \qquad
 \partial_b\log p=r_s/\sigma_s^2+r_c/\sigma_c^2-b/\tau_b^2.
\]

Retain all normalization constants in stored likelihoods and evidence calculations, even though MCMC acceptance ratios may cancel them.

NS/INS operate in normalized-prior coordinates:

\[
 u_a=\Phi(a/\tau_a),\qquad u_b=\Phi(b/\tau_b),
 \qquad u\in(0,1)^2.
\]

The transformed prior is uniform. The target in this square is `L(T(u))/Z`, not `L(T(u))*pi(T(u))/Z`. The prior transform already accounts for the prior measure. Generate open-interval uniform variates; never evaluate inverse Gaussian CDF at exactly zero or one. Any endpoint protection must have a quantified, negligible excluded prior mass.

Keep three coordinate systems distinct: physical `(a,b)`, prior-uniform `u`, and locally normalized ellipsoid coordinates used by the network. Show which is active. A rectangle of constant area in physical space generally does not have constant prior mass.

## 4. Introduction scene: why marginalization matters

**Duration:** 45–60 seconds inside the five-minute introduction, not an additional animation block.

**Initial view:** a two-channel measurement sketch and the A-linear joint posterior. Below it show the one-dimensional marginal for `a`.

**Storyboard:**

1. Move an `(a,b)` marker; update both predicted measurements and their residuals.
2. Show several different parameter pairs compatible with the signal channel.
3. Add the calibration measurement: the broad degeneracy becomes an elongated posterior.
4. Mark the best-fitting point, then show that one point does not describe uncertainty.
5. Compare `p(a | b fixed,d)` with `p(a | d)` using the calculated widths.
6. Animate contributions from horizontal slices accumulating into the marginal; preserve probability mass, not projected point brightness.

**Presenter line:** “The nuisance parameter is not the quantity I want to publish, but uncertainty about it changes the uncertainty I should publish.”

**Hierarchy bridge:** show a separate compact graph `cosmology -> latent field -> observations`, with astrophysics and instrumental parameters also entering the observation model. Say that the two-parameter example is a tractable analogue of integrating over those unknowns, not the field-level model itself.

**Do not imply:** all Bayesian inference requires sampling. This toy model is analytically solvable precisely so the samplers can be checked. Real problems may also exploit analytic marginalization and specialized integration.

## 5. Module I: independent proposals and importance weights

**Main target:** A-linear.  
**Core duration:** 60 seconds.  
**Question:** “Must the points themselves be drawn from the posterior?”

### 5.1 State and estimators

For fixed normalized proposal `q`, draw `x_i=(a_i,b_i)` independently. Define

\[
 r_i=\mathcal L(x_i)\pi(x_i)/q(x_i),\quad
 \hat Z=N^{-1}\sum_i r_i,\quad
 \bar w_i=r_i/\sum_j r_j.
\]

The weighted posterior estimate of an expectation is `sum_i wbar_i f(x_i)`. Under the usual support and integrability conditions, the fixed-proposal evidence estimator is unbiased; the self-normalized posterior ratio is generally biased at finite N. Its weight diagnostic is

\[
 N_{\rm eff,w}=1/\sum_i\bar w_i^2.
\]

This diagnostic is not a certificate of target coverage or an observable-specific MCMC ESS. [S05]

### 5.2 Frames

1. Draw an independent point from `q` and show its proposal density.
2. Evaluate its true likelihood; reveal a weight token.
3. Accumulate unweighted and weighted marginal histograms of `a`.
4. Shift or narrow the proposal using a separate preset; show unequal weights.
5. End with a few dominant points and the question: “How could we construct a better proposal?”

Marker **area**, not radius, is proportional to weight, subject to a documented display cap. Histograms and statistics always use uncapped weights. A hovering tooltip must display the actual weight.

### 5.3 Proposal presets and interaction

- `prior`: `q=pi`.
- `mismatched`: `q_fit=Normal((0.2,0.6),diag(0.35^2,0.35^2))`, mixed with the defensive prior component below.
- `pilot-fitted`: start with 512 prior proposals, use normalized likelihood weights to estimate a mean/covariance, and add `1e-4 P_0` to the covariance before fitting. Count all pilot likelihood calls. Freeze this proposal before accumulating the displayed estimation sample; do not include pilot points in that estimation sample.
- `oracle` in expert mode only: analytic A posterior. Mark it “reference proposal available only for this toy problem”; never use it to advertise a practical speedup.

Recommended defensive form: `q=(1-epsilon) q_fit+epsilon pi`, with `epsilon=0.05` in the protected preset. Since the likelihood is bounded, this also prevents unbounded likelihood/prior-to-proposal ratios arising solely from tail exclusion. This is a design choice for this teaching example, not a description of Nautilus.

**Controls:** proposal preset; discrete displacement and width; sample budget; seed. Changing the proposal starts a fresh estimation batch. Do not pool adaptively generated points with an unexplained single final proposal denominator.

### 5.4 Optional dimensionality inset

Use a separate, clearly labelled mathematical experiment:

\[
 p=\mathcal N_d(0,I),\quad q=\mathcal N_d(\delta\mathbf1,I).
\]

For fixed `d,delta`, `E_q[r]=1`, `E_q[r^2]=exp(d delta^2)`, so the large-sample weight-ESS fraction tends to `exp(-d delta^2)`. This is an analytic consequence of a fixed mismatch in every coordinate, not a universal importance-sampling scaling law. Do not substitute a two-dimensional moving picture for this calculation.

## 6. Module II: random-walk Metropolis versus HMC

**Main target:** A-linear.  
**Core duration:** 4 minutes.  
**Question:** “How should a proposal move through a correlated posterior?”

### 6.1 Common display

Use two synchronized views with identical target contours, axes, reference marginal, and initial-condition policy. A split view is used only for this direct comparison; otherwise reuse one large plot. Slow-motion algorithm steps explain mechanics, while the cost panel separately records actual numerical work.

### 6.2 Random-walk engine

Use a symmetric Gaussian proposal `x'=x+s C_q^(1/2) z` and the standard Metropolis acceptance ratio. The identity proposal is the teaching baseline. Rejection appends the current point again to the chain; it is not removed from the numerical sample. Show a visible “stay” event and repeated trace value. [S06]

**Controls:** raw-coordinate proposal widths `[0.03,0.15,0.60]`, identity versus pilot-preconditioned proposal, and seed. Whitened proposal widths are separately labelled and use `[0.3,1.0,2.0]` as candidate presets. Too-small steps and too-large steps must be real outputs of the algorithm, not hand-animated trajectories.

### 6.3 HMC engine

Use the convention

\[
 H(x,p)=-\log p(x\mid d)+\tfrac12p^T M^{-1}p,
 \qquad p\sim\mathcal N(0,M).
\]

Every iteration refreshes momentum, performs a reversible leapfrog proposal, and applies a Metropolis correction. Retain only the accepted endpoint, or the repeated current state on rejection, in this standard HMC demonstration. Intermediate leapfrog points are path vertices, **not additional posterior draws**. [S03, S06]

With this convention, a Gaussian target with covariance `C` is frequency-equalized by `M=C^-1`; equivalently transform to whitened coordinates and use unit mass. Never accidentally put the covariance where the precision is required. For a practical preconditioner use a documented pilot estimate; analytic whitening is an explicitly marked reference toggle.

**Controls:** raw-coordinate leapfrog step sizes `[0.02,0.06,0.25]`, step counts `[5,20,60]`, preconditioning, and momentum refresh/one-trajectory step. The largest raw step size is an intentionally poor candidate setting; the actual failure must be verified before replay. Whitened-coordinate candidate step sizes are `[0.1,0.3,0.8]` with counts `[3,8,20]`. All preset tuning values are proposed experimental settings and must be recorded if changed during validation. Momentum refresh is not disabled in a purported valid chain.

### 6.4 Storyboard

1. Run several Metropolis proposals slowly, including a rejection.
2. Run HMC with visible momentum and leapfrog trajectory; mark its endpoint.
3. Show accumulated samples and the marginal of `a` against the analytic reference.
4. Demonstrate a numerically unstable or poorly tuned HMC setting: display `Delta H` and its consequences rather than simply drawing a dramatic path.
5. Apply the same linear whitening information to both samplers. Acknowledge that good coordinates help random-walk MCMC too.
6. Optional 20-second switch to B-separated: local exploration may look healthy while mode occupation is wrong.

**Do not stage an unfair comparison:** a tuned HMC run versus intentionally untuned random walk can illustrate a mechanism, but cannot be labelled a performance benchmark. The conditioned Gaussian is not proof of HMC's generic superiority.

### 6.5 Diagnostics

Main display: true-likelihood call count, gradient call count, endpoint count, rejection count, and error in the estimated marginal mean/variance. The animation clock is not a runtime benchmark. Expert view may show autocorrelation and ESS for specified observables, with “insufficient chain length” when appropriate.

For B-separated, show `P(a>0)` against the exact value `0.5`; an apparently large local ESS is not evidence that both modes have been explored. Reference [S04] motivates the distinction between geometry and tuning.

## 7. Module III-A: classical nested sampling

**Targets:** begin on A-linear; switch to B-curved without changing algorithm definitions.  
**Core duration:** 3 minutes.  
**Question:** “Can we integrate over prior mass instead of wandering through the posterior?”

### 7.1 Essential distinction

The likelihood and prior must remain separate. Define

\[
 X(\lambda)=\int\mathbf1[\mathcal L(x)>\lambda]\pi(x)\,dx,
 \qquad Z=\int_0^1\mathcal L(X)\,dX.
\]

Live points represent a likelihood-constrained prior, not the posterior. Dead points acquire quadrature-based posterior weights. [S02, S07]

### 7.2 Honest baseline algorithm

For the simplest recorded teaching trace, use rejection from the **full normalized prior**, accepting a replacement only when its likelihood exceeds the current threshold. This is inefficient but transparently satisfies the constrained-prior requirement. Count every likelihood evaluation, including failed attempts.

For accelerated recordings, a validated library constrained sampler may be used, with its method, settings, and version recorded. Do not quietly swap in samples from an oracle contour. Any reference-assisted constrained draw must be marked as an oracle and excluded from performance claims.

At each iteration: remove the minimum-likelihood live point; preserve it in a dead-point ledger; obtain a valid replacement; update the estimator; record the transition. Use 128 live points for the initial teaching trace, with 32 and 512 as alternate presets. These are small visualization settings, not production recommendations. The live population size is held constant in this chapter. Dynamic NS is mentioned, not implemented as an extra lesson.

### 7.3 Prior-mass estimate and evidence

Under ideal independent constrained-prior replacements and continuous likelihood ranks, the shrinkage ratio has distribution `t_i~Beta(N_live,1)`. Use the standard log-volume expectation for the main explanation, `log X_i≈-i/N_live`, and label it an estimate. An uncertainty fan may be generated from independent shrinkage realizations, with an explicit note that this does not diagnose missed regions or correlated replacements. [S02]

Use a documented quadrature rule and include the final live contribution. For a rectangular teaching rule,

\[
 \hat Z=\sum_i\mathcal L_i(\hat X_{i-1}-\hat X_i)
       +\hat X_{\rm final}\,\overline{\mathcal L}_{\rm live}.
\]

Dead-point posterior weights are proportional to their evidence contributions; each final live point carries its corresponding share. Live points alone must never be displayed as the final posterior.

A remaining-evidence estimate based on the largest observed live likelihood is a heuristic, not a rigorous global upper bound. A toy-specific certified likelihood bound may be used only if labelled as such. [S07]

### 7.4 Frames and axes

1. Show prior samples in physical coordinates and transform the same points into the unit square.
2. Raise the likelihood threshold and discard the worst live point.
3. Show rejected and accepted replacement attempts.
4. Synchronize the shrinking allowed region with a likelihood-versus-prior-mass chart.
5. Accumulate evidence increments, then reveal the weighted posterior marginal.
6. Switch to the curved B likelihood: emphasize the cost of generating valid constrained draws.

Use a linear `X` axis when shading the area under `L(X)`. If using `t=-log X` for legibility, plot the **evidence density** `L(e^-t)e^-t`; the area under raw likelihood on a log-volume axis is not `Z`.

### 7.5 Mode-loss demonstration must identify its mechanism

Reducing the number of live points is not, by itself, a proof that a mode can never be rediscovered: full-prior rejection retains global support. A permanent mode-loss demonstration must instead identify the restrictive local bound or constrained-move mechanism that excludes it. Alternatively show finite-budget undercoverage without claiming mathematical impossibility of rediscovery.

Do not manufacture a disconnected mode failure by deleting points in the renderer. Use a recorded, labelled failed numerical run or a deliberately unsafe proposal-support control.

## 8. Module III-B: INS and the Nautilus idea

**Target:** B-curved, followed by a brief B-separated coverage check.  
**Core duration:** 3 minutes.  
**Question:** “Can previous likelihood evaluations teach us where to propose, without replacing the likelihood?”

### 8.1 Source-derived algorithm capsule

Nautilus uses INS, not classical NS quadrature. In the paper, bounds define disjoint shells; an exploration phase is followed by further sampling. For a new bound, evaluated points inside an enclosing ellipsoid provide normalized coordinates and likelihood-rank scores. Live and non-live ranks occupy the upper and lower halves of the score range. A regressor ensemble predicts these scores; a calibrated cut selects a subset of the ellipsoid for proposals. Posterior/evidence weights still use the true likelihood. Bounds need not coincide with exact iso-likelihood contours. [S01, sections 2–3 and Appendix A; S08–S09]

The 2023 paper's Figure 3 is the conceptual basis for the regression storyboard. The six operations below are a proposed animation of that mechanism, not a new inference algorithm. The figure could not be visually retrieved during this research pass; its published caption and mathematical description were cross-checked in the journal HTML.

### 8.2 Teach INS before adding the network

Use a **frozen-shell teaching estimator**. This is a controlled demonstration of the estimator, not a claim to reproduce every adaptive detail of Nautilus. Operate in uniform-prior coordinates and let the initial bound `B_0` cover the entire unit square. Define

\[
 S_i=B_i\setminus\bigcup_{k>i}B_k.
\]

The shells are disjoint even when the bounds are not strictly nested. Let `V_i` be their prior volumes and take `N_i` fresh, conditionally independent, uniform samples in each shell. Then

\[
 \hat Z=\sum_i V_i\overline{\mathcal L}_i,
 \qquad w_{ij}=\mathcal L(u_{ij})V_i/N_i,
 \qquad \bar w_{ij}=w_{ij}/\hat Z.
\]

Equivalently `g(u)=N_i/V_i` is a sample-count density in shell i, while `q(u)=g(u)/N_total` is normalized. Consequently `sum w`, not the average of these already count-normalized weights, estimates evidence. This is also the ordinary fixed-proposal formula `L/(N_total q)` applied to a stratified design. [S01; mathematical specification and frozen-design derivation]

**Boundary conditions:** finite `V_i`, full target support, correct region membership, and uniform sampling within frozen shells. Numerical volume uncertainty must be measured and propagated. For the frozen teaching estimator, estimate volumes with an independent geometry-only random stream, separate from the likelihood-estimation samples; retain covariance between partition-volume estimates. The production replay uses library-returned weights, not this formula pasted over an arbitrary adaptive point cloud.

### 8.3 INS storyboard

1. Return to the importance-sampling weight token from Module I.
2. Show several proposal bounds laid over the curved region.
3. Partition their union using the shell definition, with the outer shell completing the prior.
4. Draw at different densities in different shells; show that raw point density is not posterior density.
5. Apply shell-dependent weights; accumulate `V_i mean(L)_i` and the posterior marginal.
6. Add samples to a selected shell and update its count-dependent weights, including weights of earlier retained samples in that shell.

Do not enforce exact likelihood thresholds on fresh INS estimation samples. A likelihood evaluation below the current live threshold can still inform the INS estimator. Distinguish this from a neural-filter rejection before the true likelihood was evaluated: the latter is not a posterior-weighted point.

### 8.4 Regression storyboard

Use one large canvas with progressive overlays, not a dense six-panel diagram.

**Frame N1 — evaluated information.** Show points already evaluated by the sampler, including those outside the current live set. The reference contour may be visible to the audience but must not enter training.

**Frame N2 — geometric envelope.** Draw an enclosing ellipsoid or a documented union of proposal ellipsoids. Mark its geometrically permitted candidate region.

**Frame N3 — learning coordinates and targets.** Briefly transform the points to local ellipsoid coordinates. The training target is a likelihood-rank score, not a posterior weight or an independent probability of correctness. Use the paper's split-rank definition:

\[
 s_{\mathcal L}=\begin{cases}
 \tfrac12 r_{\rm nonlive},&\text{non-live},\\
 \tfrac12(1+r_{\rm live}),&\text{live}.
 \end{cases}
\]

Ranks are fractions within the respective sets. Tie handling must be deterministic and documented. Export the exact calibration from the chosen implementation; do not silently assume that the cut on predicted score is always `0.5`.

**Frame N4 — regression, not network architecture.** Show the map `local coordinates -> predicted likelihood score`. Use a scatter plot of actual versus predicted scores and a simple regressor icon. The predicted score is not necessarily bounded to `[0,1]` even when the training targets are. Do not call the ensemble's spread a calibrated posterior uncertainty.

**Frame N5 — learned proposal region.** Overlay the accepted-score region and compare it with the geometric envelope. For a single frozen ellipsoid, its volume is the ellipsoid volume times the measured neural acceptance fraction. For an overlapping union, apply the proper overlap treatment before estimating volume. [S01, S11]

**Frame N6 — true likelihood and weights.** Send surviving candidates to the true likelihood. Show the likelihood-call counter advancing only there. Send evaluated points, including low-likelihood evaluations, to the estimator with the appropriate weights. Neural rejects have no true-likelihood value and receive no estimator contribution.

Keep a short on-screen sentence: **“Learn where to propose; evaluate what is true.”** This is a design slogan, not a claim of unconditional correctness.

### 8.5 Controls, labels, and ablation

Main controls: geometric envelope / learned-region overlay; trace snapshot; candidate replay; seed/preset. In expert mode expose score cut and proposal acceptance, but label this a **proposal-only sensitivity view**. Altering a cut must not leave old posterior/evidence diagnostics on screen as if they remained valid.

For a clean local ablation, freeze the same region and candidate bank. Compare how many true-likelihood calls a geometric-only versus learned-filter proposal would make. This demonstrates selection cost and filtering errors at that snapshot. It is **not** an end-to-end performance comparison of two valid samplers. Export actual full-run comparisons separately if they are later computed.

At least one recorded run must come from the actual Nautilus package with its version, full settings, RNG seed, and provenance saved. Use explicit settings rather than undocumented defaults. Suggested initial research settings are `n_live=2000`, `n_networks=4`, `n_eff=10000`, and `discard_exploration=True`, subject to convergence validation. These are proposed settings, not a guarantee of success or an assertion of current defaults. [S11]

If the implementation instead uses a reduced teaching estimator, its permanent title must say **“Nautilus-inspired INS demonstration”**. Do not present this as a verified replay of production Nautilus.

### 8.6 Required caveats

The official documentation recommends discarding exploration samples for publication estimates to avoid pseudo-importance bias. For the teaching estimator, freeze bounds and draw fresh estimation samples. For production replay, respect the package's weighting/reuse rules; never naively pool all adaptive candidates. [S09]

Support in principle is not adequate sampling in practice. A learned region can underallocate an important mode so severely that finite runs miss it. Importance weights cannot recover likelihood information from a region never sampled. Compare seeds, budgets, and live-point counts; do not label a completed stopping condition as a convergence guarantee. [S09]

The current FAQ discusses successful applications with tens of dimensions and limitations at higher dimensions. Do not advertise this method as an established million-dimensional field sampler. The documentation also describes CPU-based networks; this presentation must not imply that the reference implementation is GPU-native. [S08]

## 9. Optional Example C: hierarchical geometry

Use only in discussion or the 25-minute route. Define

\[
 \lambda\sim\mathcal N(0,2^2),\quad
 z\mid\lambda\sim\mathcal N(0,e^\lambda),\quad
 y\mid z\sim\mathcal N(z,\sigma_y^2),
\]

with `y=0.2` and weak observation `sigma_y=3`. Here `lambda` is log population variance and `z` a latent source/field amplitude. This is a genuine hierarchy, albeit tiny.

Compare centered `(lambda,z)` and non-centered `(lambda,xi)` with `z=exp(lambda/2)xi`, `xi~Normal(0,1)`. Verify the Jacobian relation explicitly. Reparameterization changes computational geometry, not the physical posterior. With weak data it can help; with strong data it need not be the better parameterization. [S04]

Reference evidence is the one-dimensional integral

\[
 Z=\int\pi(\lambda)\mathcal N(y;0,e^\lambda+\sigma_y^2)\,d\lambda.
\]

In the no-data limit this becomes a hierarchical **prior** that can be sampled ancestrally. Do not present that special limit as a posterior requiring MCMC.

**Discussion question:** “When the sampler is slow, should we change the sampler, the coordinates, or the variables we explicitly sample?”

## 10. Presentation routes

### 10.1 Strict 20-minute route

| Time | Content |
|---|---|
| 0:00–5:00 | Inverse problems, hierarchy, and A marginalization scene |
| 5:00–6:00 | Module I: importance sampling |
| 6:00–10:00 | Module II: random walk and HMC |
| 10:00–13:00 | Module III-A: classical nested sampling |
| 13:00–16:00 | Module III-B: INS and learned bounds |
| 16:00–19:00 | Future directions, using frozen final animation states |
| 19:00–20:00 | Audience question / buffer |

Only one live intervention per module is essential. Use rehearsed seeded presets for everything else. Do not attempt to expose every expert control in the main talk.

### 10.2 25-minute route

Keep the opening at five minutes. Add one minute for fair preconditioning, one minute for B-separated mode coverage, two minutes for Example C, and one minute for questions. Do not add a fourth full sampler tutorial.

### 10.3 Future-looking conclusions tied to existing scenes

- **Geometry:** the preconditioning/funnel state asks how physics-informed coordinates and partial marginalization can help.
- **Expensive evaluations:** the learned-bound state asks which cheap computations can avoid expensive likelihood calls without silently changing the target.
- **Parallel hardware:** show an optional task timeline for independent importance proposals, sequential Markov transitions, independent chains, and batched INS likelihood calls. Distinguish concurrency from less total work.
- **Reliability:** the mode-coverage state asks how to detect an apparently convincing but wrong posterior.

These are research opportunities, not forecasts of a universally winning algorithm. ML remains one concrete proposal-construction example, not the organizing taxonomy of the talk.

## 11. Interaction and rendering specification

### 11.1 Layout

Design for 1280x720 logical pixels, scaling to a 1920x1080 presentation display. Main labels should be legible at projection distance: approximately 22–28 logical pixels, with larger titles. Keep technical formulas in a single changing caption rather than many permanent boxes.

Default layout: target/problem canvas about 60–65% width; one supporting diagnostic about 25–30%; a narrow footer for algorithm state, computational counts, and controls. Use shape/line-style differences as well as color. No reliance on red-versus-green alone. Stable positions and styles identify accepted points, rejected proposals, live/dead points, and weighted points.

### 11.2 Controls

Space = play/pause; right arrow = one meaningful algorithm event; left arrow = previous recorded event; R = reset current seeded trace; numbers select main modules. Include clickable equivalents and a reduced-motion mode. Do not advance a stochastic engine backward by reversing its random-number generator; rewind the recorded ledger.

Presenter mode offers a small number of named presets. Expert mode exposes counts, provenance, coordinates, tuning, and validation data. Parameter controls that select prerecorded configurations must visibly snap to available values. Do not imply that a smoothly animated interpolation is a newly computed run.

Separate **target controls** from **sampler controls**. Any change to data, likelihood, prior, or coordinate-density definition invalidates accumulated inference diagnostics. Tuning changes either select an independent complete trace or start an explicitly marked warmup/reset stage.

### 11.3 Safe replay and fallback

Preferred architecture: genuine offline numerical runs -> versioned event/diagnostic files -> deterministic browser replay. This minimizes live numerical and neural-training risk while preserving play, pause, step, compare, and preset selection.

The application must start and replay with networking disabled. Bundle its assets and trace data locally; avoid external fonts, remote plotting APIs, CDN-only math rendering, and dependence on an online notebook kernel. A future build may package a local server, but a static presentation route is preferred.

Provide one MP4 fallback per principal module plus a contact sheet or PDF of key states when implementation is commissioned. These are required future build outputs, not artifacts supplied with this specification. Statistical counters must come from the same trace as the animation.

## 12. Scientific and software boundaries

### 12.1 Components

1. **Model definitions:** normalized priors/likelihoods, gradients, prior transforms, immutable preset identifiers.
2. **Reference calculators:** analytic Gaussian results and independent quadrature; hidden from sampler training and charged separately from algorithm work.
3. **Sampler/trace producers:** importance, RWM, HMC, classical NS, and actual Nautilus replay adapter or clearly labelled teaching INS.
4. **Diagnostics:** observable estimates, counts, validity flags, and run-level reference comparisons.
5. **Renderer/player:** consumes traces; cannot invent accepted points, weights, evidence, or training data.
6. **Presenter state machine:** module transitions, pause points, explanatory overlays, and timing.

A module can be understood and tested without reading unrelated modules. Shared numerical definitions are not copied into independent untested scripts.

### 12.2 Trace schema requirements

Every trace contains:

- `schema_version`, `model_id`, model parameter values and hash;
- algorithm name, implementation/version, complete settings, seed and RNG identity;
- actual versus pedagogical/oracle mode;
- physical and prior coordinates as appropriate;
- event type, likelihood, prior, proposal density/weight where defined;
- chain state versus proposal/intermediate path status;
- live/dead IDs or bound/shell IDs;
- phase: warmup, exploration, frozen estimation, or production sampling;
- counts of true likelihood calls, gradient calls, neural predictions, geometric candidates, and training operations;
- posterior/evidence diagnostics with explicit validity flags;
- display subsampling rules and sampled-point identifiers;
- cumulative measured computational time where available, separate from playback time.

For a network snapshot, also save training-point IDs, actual score targets, normalization transforms, calibrated cut, fit settings, and the frozen predicted-score grid or model parameters needed for replay. An unseen true-likelihood grid must not be passed to the regressor.

### 12.3 Backend implementation policy

Do not promise a production Nautilus callback or undocumented `n_networks=0` ablation flag. First inspect the pinned release's public methods and returned fields. If tracing needs an adapter or instrumentation, verify that it does not alter numerical results. Public outputs such as posterior points, weights, log likelihoods, and evidence should be cross-checked against the uninstrumented run. [S11]

Do not create a new software project or choose final dependency versions solely from this document. Confirm the existing repository/platform before implementation; the mathematical and interaction contracts above do not depend on a particular frontend framework.

## 13. Numerical validity and fair comparisons

### 13.1 Mandatory invariants

- Every plotted posterior is associated with a normalized prior and fixed likelihood.
- `log L` used for evidence retains normalization constants.
- RWM and HMC rejected iterations remain in the chain.
- HMC path vertices are not automatically posterior samples.
- NS live points are constrained-prior samples, not posterior samples.
- Classical NS and INS use their own estimators; toggling a neural overlay does not convert one into the other.
- Neural predictions choose proposals; exact likelihood evaluations determine final weights.
- Overlapping proposal regions do not double-count probability mass.
- INS estimates use a valid adaptive scheme or fresh sampling from a frozen design.
- An external display/reference calculation is never misreported as a free sampler capability.
- High estimated ESS cannot override failed mode coverage or other reference tests.

### 13.2 Accuracy metrics

The common comparison currency is accuracy in specified scientific quantities versus computational resources, not a universal ESS number.

For A: mean and standard deviation of `a`, covariance, marginal CDF discrepancy, and `log Z` for methods that estimate it. For B: `P(a>0)`, `E[a^2]`, marginal CDFs, and `log Z`. A mean of zero alone cannot validate B, since a distribution with the wrong shape can have that mean.

Report importance-weight ESS, MCMC observable-wise ESS, and NS uncertainty diagnostics with different labels. For stochastic benchmark comparisons use multiple seeds and estimates of Monte Carlo error. Ordinary HMC does not intrinsically output evidence; show a dash rather than an invented evidence estimate.

### 13.3 Computational accounting

Maintain separate counters for likelihood evaluations, gradient evaluations, and overhead. An HMC gradient is not assumed to cost one likelihood call. A combined value-and-gradient evaluation must be recorded as one combined operation, not billed twice as two full model runs. If a cost model combines them, display the chosen conversion factors and label the result a cost-model estimate.

A measured wall-clock comparison must include pilot tuning and network training and specify hardware, parallelism, and implementations. A likelihood-budget comparison is different from a runtime comparison. For cheap toy likelihoods, neural overhead may dominate; a lower likelihood-call count is not automatically a faster wall time. [S03, S08, S12]

Parallelism inset, if built: label its worker count and simulated per-task times. Do not show idealized scheduler bars as a measured scaling law. Resampling or bound updates introduce global work; multiple MCMC chains can also run independently.

## 14. Verification and acceptance criteria

These are requirements for the future implementation, not completed tests.

### 14.1 Deterministic mathematical tests

| ID | Test | Acceptance |
|---|---|---|
| V01 | Analytic A evidence/moments versus independent 1D quadrature | Relative agreement <=1e-8 for evidence and moments with stable absolute tolerances near zero |
| V02 | B quadrature convergence | Tightening tolerances and splitting both modes changes log Z by <1e-6 and reported moments by <1e-5 |
| V03 | Analytic gradients versus centered finite differences | Relative error <1e-6 away from underflow and extreme tails |
| V04 | Prior transform/inverse and density change | Roundtrip errors within float precision on ordinary points; tail exclusion documented |
| V05 | B sign symmetry | Reference P(a>0)=0.5 within quadrature error |
| V06 | All weights / shell partition | No overlaps in shell assignment; full prior coverage; normalized posterior weights sum to one |
| V07 | Frozen INS constant-likelihood case | Evidence equals that constant within independently quantified volume error |
| V08 | Evidence normalization | Multiplying L by known positive c shifts log Z by log c without changing normalized posterior |

### 14.2 Sampler-specific tests

**RWM:** confirm rejected transitions duplicate the previous state; run analytic A checks after recorded warmup; include symmetric-proposal and transformed-density tests.

**HMC:** test reversibility to numerical tolerance; verify decreasing energy error as leapfrog step size is reduced at fixed integration time; test acceptance using full Hamiltonian; compare endpoints to analytic A. Record Gaussian covariance/precision convention in a unit test. Reject non-finite trajectories visibly rather than silently dropping them.

**Classical NS:** validate constrained draws in a simple bounded/uniform test problem; examine shrinkage/insertion ranks or an appropriate independent check; verify final-live contribution and repeat-seed evidence behavior. A low-dimensional reference test must expose a deliberately unsafe constrained sampler. [S02, S13]

**Frozen INS:** deliberately use non-nested overlapping bounds, partition them by set difference, and compare evidence against independent quadrature. Test that naive overlap double-counting fails. Repeat with a poor but full-support proposal; efficiency can degrade while the estimator remains consistent. Test all shell count updates.

**Nautilus replay:** compare instrumented and uninstrumented outputs; retain version/seed/settings; use multiple live-point settings and repeat runs; compare production weights to the exported ledger. Demonstrate `discard_exploration` behavior without equating it to a universal guarantee. [S09, S11]

**Network:** training-point ledger excludes all reference-only evaluations. Permuting score labels must degrade its predictive usefulness. The UI must still distinguish true and predicted scores. Candidate rejection occurs before likelihood evaluation. A missing-mode or over-tight-region stress test must trigger a coverage warning or visibly wrong reference statistic, not a reassuring success message.

### 14.3 Stochastic checks

Use at least 20 independent seeds for normal acceptance runs, with a 100-seed confirmation for suspicious evidence bias or mode-coverage failures. Report the distribution of errors, not only the most attractive animation seed. Displayed failure seeds may be deliberately illustrative, but label them and do not use them to estimate failure frequencies.

For a benchmark estimate of mean error, require agreement with zero within a stated multiple of its across-run standard error and a practical tolerance appropriate to the displayed precision. If not, increase budget or revise settings; do not loosen tests after seeing results without documenting why. Posterior coverage experiments require newly generated datasets, not just many seeds for the same fixed dataset. Use the term “reference recovery” for the latter.

### 14.4 Presentation acceptance

- Complete the 20-minute route without live coding or an online service.
- Every animation can pause at its key conceptual frame and advance one meaningful event.
- No chart treats playback speed as scientific speed.
- The right posterior or evidence remains attached to the correct preset after every control change.
- All scene labels remain legible at 16:9 projection size.
- Replays are deterministic for the saved seed/run.
- Missing assets or an unavailable preset produce an explicit message, not an interpolated fake run.
- Video fallback and static final-state view preserve the same estimators and counters.

## 15. Build sequence and review gates

This is a design-level work breakdown, not an implementation authorization.

**Stage 1 — shared models and references:** implement and test A/B definitions, priors, transforms, gradients, and quadrature. Freeze model identifiers and reference values.

**Stage 2 — core sampler traces:** importance sampling, RWM, and HMC; establish event schema, accounting, and independent diagnostics before rendering.

**Stage 3 — classical NS:** first use a transparent valid constrained-prior baseline, then record any accelerated implementation separately. Validate volume/evidence accounting.

**Stage 4 — frozen INS and Nautilus reference:** verify partition weights independently; obtain one genuine instrumented Nautilus trace; add learned-bound snapshots and the proposal-only ablation.

**Stage 5 — presentation application:** deterministic replay, shared axes, presenter controls, main timing route, and offline packaging.

**Stage 6 — statistical and visual review:** multiple-seed checks, failure cases, instrumentation equivalence, readability, and rehearsal. Export fallbacks only after the numerical content is fixed.

**Stage 7 — optional work:** Example C, dimensionality inset, parallel worker timeline, and live in-browser recomputation. These cannot delay the three principal demonstrations.

Written-spec approval should precede a repository-specific executable implementation plan. No code, dependencies, deployment, or external project was created as part of this specification.

## 16. Sources and provenance

Source references identify established algorithmic facts. The particular two-channel toy model, numerical presets, scene grouping, UI, validation thresholds, timing, and software decomposition are design proposals developed here; they are not claims made by the Nautilus paper.

- **[S01]** Lange (2023), *Nautilus: boosting Bayesian importance nested sampling with deep learning*, arXiv:2306.16923. Sections 2–3 and Appendix A are the principal algorithmic source. URL: `https://arxiv.org/pdf/2306.16923`. Published HTML cross-check: `https://academic.oup.com/mnras/article/525/2/3181/7243406`.
- **[S02]** Ashton et al. (2022), *Nested sampling for physical scientists*, arXiv:2205.15570. URL: `https://arxiv.org/abs/2205.15570`.
- **[S03]** Stan Reference Manual, *MCMC Sampling*. URL: `https://mc-stan.org/docs/reference-manual/mcmc.html`.
- **[S04]** Stan User's Guide, *Efficiency Tuning*, especially reparameterization and hierarchical models. URL: `https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html`.
- **[S05]** Elvira, Martino & Robert, *Rethinking the Effective Sample Size*, arXiv:1809.04129. URL: `https://arxiv.org/abs/1809.04129`. The fixed-proposal estimator identities in this design are direct mathematical derivations.
- **[S06]** Neal, *MCMC using Hamiltonian dynamics*, arXiv:1206.1901. URL: `https://arxiv.org/abs/1206.1901`.
- **[S07]** dynesty documentation, *Background*. URL: `https://dynesty.readthedocs.io/en/stable/overview.html`.
- **[S08]** Nautilus documentation, *FAQs*: role of the emulator, CPU computation, dimensionality limits, weighted outputs. URL: `https://nautilus-sampler.readthedocs.io/en/stable/discussion/faqs.html`.
- **[S09]** Nautilus documentation, *Convergence*: sampling support, stopping versus convergence, pseudo-importance bias. URL: `https://nautilus-sampler.readthedocs.io/en/stable/discussion/convergence.html`.
- **[S10]** Nautilus documentation, *Nested Sampling*: distinction between NS and INS. URL: `https://nautilus-sampler.readthedocs.io/en/stable/discussion/nested_sampling.html`.
- **[S11]** Nautilus full API documentation. URL: `https://nautilus-sampler.readthedocs.io/en/stable/api_full.html`.
- **[S12]** Nautilus documentation, *Parallelization*. URL: `https://nautilus-sampler.readthedocs.io/en/stable/guides/parallelization.html`.
- **[S13]** Buchner, *A statistical test for Nested Sampling algorithms*, arXiv:1407.5459. URL: `https://arxiv.org/abs/1407.5459`.

**Research limitation:** the arXiv PDF text and journal HTML were readable, but web screenshots and direct figure-image retrieval failed. The specification therefore uses the paper's mathematics and figure captions; it does not claim independent visual inspection of the original figure artwork or remeasurement of plotted performance. No performance numbers from its figures are needed for the design.
