# Research and design notes
## Shared inference problems, nested sampling, and the Nautilus regression idea

20 October 2026. Companion to `2026-10-20-sampling-animations-design.md`.

## 1. What the research changes

The earlier outline was organized mainly by algorithm, with a generic common posterior. The revised design organizes the lesson around a measurement problem and two increasingly difficult versions of its response. This preserves comparisons while keeping likelihood, prior, posterior density, and prior mass distinct.

The principal correction concerns Nautilus: adding a neural-region overlay to a classical nested-sampling animation is not enough. The audience needs to see a change of estimator from shrinking-prior-mass quadrature to importance weighting. The regression component then improves proposals rather than supplying a replacement posterior. The official documentation explicitly distinguishes NS from INS. [R1]

Consequently, the animation's conceptual branches are:

- independent proposals and weights -> geometry-dependent importance efficiency;
- random-walk proposals -> Hamiltonian proposals -> parameterization;
- constrained-prior exploration -> evidence quadrature;
- importance weighting plus learned proposal regions -> accurate likelihood evaluations with nontrivial coverage requirements.

The network is a modification to proposal construction, not a fourth philosophical foundation of Bayesian sampling.

## 2. Why two examples, not one or five

**One generic contour is insufficient.** It can hide whether the plotted object is likelihood or posterior. The distinction matters especially when one wants to introduce evidence. It also makes a neural network look unnecessary if the chosen distribution is exactly elliptical.

**Several real cosmological likelihoods are excessive.** In a 20-minute talk, repeatedly explaining new data, parameters, and selection effects would compete with the numerical content. The user wants algorithms, intuition, and research opportunities, not a sequence of application mini-talks.

**The selected compromise is a two-channel synthetic measurement:** a signal contaminated by an unknown additive background and an independent measurement of that background. A linear response gives an analytic, highly correlated posterior. A squared response gives curved degeneracy and two possible signs. Calibration strength separates or connects their high-likelihood neighborhoods.

The models were designed for this presentation. They are not a reproduction of a Nautilus paper benchmark or a published weak-lensing analysis. Their usefulness is that one nuisance parameter can be integrated out analytically, leaving an independent one-dimensional reference integral even for the nonlinear case.

## 3. Source-derived Nautilus capsule

The paper learns likelihood-rank scores from evaluated points in local bounding-ellipsoid coordinates, using live and non-live points. The regression identifies a smaller proposal region; it does not provide final importance weights. Its shell construction also accommodates bounds that are not perfectly nested. The final design follows the mathematical description and Figure 3 caption, not a neuron-by-neuron visualization. [R2]

The documentation clarifies that actual posterior and evidence estimates use true likelihoods. It also warns that the reference implementation is CPU-based and its demonstrated dimensional regime is far below millions of latent variables. [R3]

**Design consequence:** three counters must remain distinct: cheap proposal candidates, neural predictions, and expensive true-likelihood evaluations. Points screened out before evaluating the likelihood are not part of a weighted posterior sample. Evaluated points that fail a live-likelihood threshold are a different category.

## 4. What must not be simplified away

### 4.1 Ordinary NS uses prior mass

The mathematical quantity is `X(lambda)=integral 1[L>lambda] pi`, not visual area in arbitrary coordinates. Normalized-prior coordinates make the area interpretation legitimate. The same point cloud can therefore be animated through a prior transform before the NS lesson. Its final posterior comes from weighted evidence contributions, not the last live cloud. [R4]

**Design consequence:** likelihood and posterior overlays have separate legends. If the evidence chart uses `-log X`, its plotted integrand includes the Jacobian. The renderer must not shade an area that is mathematically different from evidence.

### 4.2 Reusing adaptive samples requires correct accounting

The convergence documentation distinguishes a stopping condition from actually obtaining reliable results. It recommends varying settings and, for publication results, discarding exploration samples to remove the pseudo-importance-bias contribution. [R5]

**Design consequence:** the teaching INS estimator freezes a partition and draws fresh points. A production Nautilus replay uses library weights and records its exploration-discard setting. “We keep every point, so the answer is exact” is prohibited language.

The frozen partition is also a useful way to teach why weights work. Once the geometry is fixed, `Z=sum_i V_i mean(L)_i` follows directly from partitioning the integral. The construction can be demonstrated without requiring the audience to understand every adaptive detail in Appendix A.

### 4.3 Efficient weights do not prove coverage

The inverse sum of squared normalized weights is a useful concentration diagnostic, but it is not a universal approximation to every relevant effective sample size. The methodological literature specifically cautions against interpreting it too broadly. [R6]

**Design consequence:** use an exact observable such as the probability of the positive-amplitude solution, not just a reassuring ESS counter. Repeat seeds and settings in the scientific validation, even if the presentation replays only one illustrative run.

### 4.4 HMC is not a gradient-descent animation

HMC introduces momentum, follows an approximate Hamiltonian trajectory, and corrects the proposal with an acceptance step. The conventional endpoint algorithm does not treat every displayed integrator vertex as a sample. [R7]

**Design consequence:** path vertices are transient line markers; posterior draws have a different symbol. Rejected proposals leave a repeated chain state. Numerical work includes gradient evaluations, not just accepted endpoints.

### 4.5 Parameterization should not be an unfair advantage

The geometry of hierarchical models may depend strongly on coordinates; non-centered parameterizations can help in weak-data regimes but are not automatically best with informative data. [R8]

**Design consequence:** whitening is shown as a shared opportunity for random-walk MCMC and HMC, with a distinct backup hierarchy illustrating variable curvature. An analytic covariance is an oracle reference, not a free capability credited to one practical sampler.

## 5. Regression versus other ML opportunities

The relevant lesson is narrow: previous expensive evaluations can train a cheap selection rule. There is no need to teach conditional density estimation, normalizing flows, diffusion sampling, or amortized posterior inference.

The teaching UI should expose the input/output relation and filtering consequences, not a network architecture diagram. A scatter plot of actual versus predicted score is useful because it makes the difference between a score and a likelihood visible. The learned threshold should come from the run, rather than a generic classifier convention.

A neural-region control has two meanings that must not be conflated. A proposal-only sensitivity control can show what points a tighter region would filter. But a posterior result under that new region requires a new valid run or a carefully reweighted sampling design. Old posterior error bars cannot remain attached to a modified proposal as if the computation had been redone.

## 6. Practical architecture decision

### Options

1. **Everything live in the browser:** flexible, but requires a trustworthy implementation of neural training, region volumes, adaptive accounting, and samplers under a presentation deadline.
2. **Only rendered movies:** robust, but does not support stepping, alternate seeds, or meaningful diagnostic controls.
3. **Verified numerical traces plus interactive replay:** supports genuine algorithm events and presenter controls without live-training risk.

The specification chooses option 3. This is a new recommendation, not a remembered user platform requirement.

Real implementations provide additional fields useful for a provenance-preserving trace, and Nautilus documents separate sampling and posterior-output operations. But the exact adapter must be based on a pinned release, not an assumed callback API. The documentation inspected during this research contains an example of why defaults should be explicit: the sampler signature and nearby prose list different live-point defaults. [R9]

The primary animation should therefore not depend on a guessed package flag for “turn off all networks.” A fixed-snapshot proposal ablation is easier to define honestly; a full-run performance ablation requires separately validated implementations and budgets.

## 7. Completed mathematical checks

These checks validate the example definitions, not the sampling animations.

For `d=(1,0)`, signal noise `0.12`, and independent prior widths `2` and `1`, analytic Gaussian evidence and an independently arranged one-dimensional quadrature agree to within floating-point precision on both linear calibration presets.

The weak-calibration linear case has posterior correlation about `-0.9721`. Its marginal uncertainty in signal amplitude is about `0.5108`, compared with conditional uncertainty `0.1198`. This is why it is a useful opening example: the difference between conditioning and marginalizing is large and exactly calculable.

The nonlinear presets give a probability of `0.5` for each sign, as required by symmetry. Their evidence and moments are included in `reference_checks.json`. The two signs remain part of the same model; posterior mode probabilities must not be mislabeled as Bayes factors between separate models.

No RWM, HMC, NS, INS, or Nautilus performance comparison was run. No network was trained. Animation frame timing and visual readability have not yet been tested in a built application.

## 8. Limits of the research pass

The arXiv PDF text and the journal's HTML mathematical discussion were available. Attempts to retrieve PDF page screenshots and the journal figure image failed. Figure-based planning therefore relies on the published captions and surrounding mathematical description. No claim is made that the original artwork was visually inspected. No performance number was read off a chart.

The original attached Garaldi slides were already available in the conversation and establish continuity with the differentiable-simulation discussion. This design does not claim to have refreshed the forum calendar or confirmed a changed presentation slot. It accommodates the user's 20-minute framing and the previously discussed five-minute extension.

## 9. Source index

All sources below are primary papers, official documentation, or an author's methodological review. Unofficial AI-generated paper summaries were not used as evidence.

- **[R1]** Nautilus documentation, *Nested Sampling*. `https://nautilus-sampler.readthedocs.io/en/stable/discussion/nested_sampling.html`
- **[R2]** Lange (2023), *Nautilus: boosting Bayesian importance nested sampling with deep learning*, arXiv:2306.16923, sections 2–3 and Appendix A. `https://arxiv.org/pdf/2306.16923`; journal cross-check `https://academic.oup.com/mnras/article/525/2/3181/7243406`
- **[R3]** Nautilus documentation, *FAQs*. `https://nautilus-sampler.readthedocs.io/en/stable/discussion/faqs.html`
- **[R4]** dynesty documentation, *Background*. `https://dynesty.readthedocs.io/en/stable/overview.html`; Ashton et al. (2022), *Nested sampling for physical scientists*. `https://arxiv.org/abs/2205.15570`
- **[R5]** Nautilus documentation, *Convergence*. `https://nautilus-sampler.readthedocs.io/en/stable/discussion/convergence.html`
- **[R6]** Elvira, Martino & Robert, *Rethinking the Effective Sample Size*, arXiv:1809.04129. `https://arxiv.org/abs/1809.04129`
- **[R7]** Neal, *MCMC using Hamiltonian dynamics*, arXiv:1206.1901. `https://arxiv.org/abs/1206.1901`; Stan Reference Manual, *MCMC Sampling*. `https://mc-stan.org/docs/reference-manual/mcmc.html`
- **[R8]** Stan User's Guide, *Efficiency Tuning*. `https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html`
- **[R9]** Nautilus full API documentation. `https://nautilus-sampler.readthedocs.io/en/stable/api_full.html`
- **[R10]** Nautilus documentation, *Parallelization*. `https://nautilus-sampler.readthedocs.io/en/stable/guides/parallelization.html`
- **[R11]** Buchner, *A statistical test for Nested Sampling algorithms*. `https://arxiv.org/abs/1407.5459`
