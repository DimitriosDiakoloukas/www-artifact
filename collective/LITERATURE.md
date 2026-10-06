# Novelty boundary for the collective-dependence study
Verified against primary sources on 5 October 2026 UTC. This is an initial boundary check, not a complete
literature review or a priority claim. Mathematical conditional-variance identities and Hoeffding bounds
are standard. General GNN range, explanation fidelity and sufficiency are already established topics.

- Jacob Bamberger, Benjamin Gutteridge, Scott Le Roux, Michael M. Bronstein and Xiaowen Dong.
  *On Measuring Long-Range Interactions in Graph Neural Networks*. ICML 2025, PMLR 267:2770–2789.
  [Publisher](https://proceedings.mlr.press/v267/bamberger25a.html).
  Formalizes long-range interactions, provides a graph-operator range measure, and validates it on
  synthetic and real tasks. The new study cannot claim to introduce graph-range measurement or its
  synthetic validation. It investigates binary signed inputs, collective distant completions, and
  disagreements between individual and collective diagnostics under known dependency structures.

- Kenza Amara, Zhitao Ying, Zitao Zhang, Zhichao Han, Yang Zhao, Yinan Shan, Ulrik Brandes,
  Sebastian Schemm and Ce Zhang. *GraphFramEx: Towards Systematic Evaluation of Explainability Methods
  for Graph Neural Networks*. Learning on Graphs 2022, PMLR 198:44:1–44:23.
  [Publisher](https://proceedings.mlr.press/v198/amara22a.html).
  Evaluates sufficient and necessary explanations with fidelity and a characterization score, including
  an eBay fraud case study. The proposed diagnostic must be differentiated from established explanation
  fidelity; a radius label alone does not create novelty. Our conditional variance measures the binary
  prediction’s variability under an explicit completion distribution and distance-based conditioning.
  Its value still requires empirical validation and a useful computation decision.

- Ruichu Cai, Yuxuan Zhu, Xuexin Chen, Yuan Fang, Min Wu, Jie Qiao and Zhifeng Hao.
  *On the Probability of Necessity and Sufficiency of Explaining Graph Neural Networks:
  A Lower Bound Optimization Approach*. arXiv:2212.07056v4, 29 December 2024.
  [Author manuscript](https://arxiv.org/abs/2212.07056v4).
  NSEG optimizes a lower bound on the probability of necessity and sufficiency using an SCM and
  counterfactual estimation. Joint interventions and necessity/sufficiency are not new concepts.
  Our stress distributions do not establish causal effects on users or make SCM-based necessity claims.

- Sayan Saha and Sanghamitra Bandyopadhyay. *Certified Evaluation of Model-Level Explanations for
  Graph Neural Networks*. ICLR 2026.
  [Publisher](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c354d2a64a71776538d6dbbc38285216-Abstract-Conference.html).
  Introduces sufficiency risk and distribution-free certificates for model-level class motifs, with
  Coverage, Greedy Gain Area and Overlap and finite-sample concentration bounds. The primary proceedings
  verify authors and publication status even though direct OpenReview retrieval presents a browser
  challenge. The full technical comparison remains necessary before any priority claim. Our MC intervals
  concern variability under an explicitly selected completion Q; they do not certify model-level
  explanation sufficiency or validate Q.

The intended contribution is an experimentally supported account of when individual signed sensitivity
understates collective dependence, a practical estimator under explicit assumptions, and a demonstrated
computation-allocation decision. Only the pilot foundation is complete at this stage. No "first", "new
sufficiency theorem", causal or deployment claim is justified by the present results.

## Variance-based sensitivity analysis is a direct mathematical predecessor
- I. M. Sobol'. *Global sensitivity indices for nonlinear mathematical models and their Monte Carlo
  estimates*. Mathematics and Computers in Simulation 55(1–3):271–280, 2001.
  DOI: 10.1016/S0378-4754(00)00270-6.
  [Publisher](https://www.sciencedirect.com/science/article/pii/S0378475400002706).
  Defines global indices for individual inputs and groups and Monte Carlo estimates of total variance.
  Under an independent product Q, our E Var(p | near signs) is precisely the **unnormalised total-effect
  variance of the distant input group**. This follows by identifying the distant group with Sobol's
  input subset and the near group with its complement. The paired estimator belongs to established
  pick-and-freeze sensitivity analysis; neither the variance quantity nor paired completion alone can
  be presented as a new general method. The potential contribution is their distance-specific signed
  application, validation under known independent and correlated dependency regimes, and practical use.

- Art B. Owen. *Sobol' Indices and Shapley Value*. SIAM/ASA Journal on Uncertainty Quantification
  2(1):245–251, 2014. DOI: 10.1137/130936233.
  [Publisher](https://epubs.siam.org/doi/10.1137/130936233).
  Studies variance-based attribution of input variables and relates it to Shapley values. The general
  distinction between large random changes and local derivatives also precedes this pilot.

This direct predecessor strengthens the novelty constraint in the original plan. A compelling paper
must establish new empirical or mechanistic knowledge and a useful decision, rather than rename a
standard total-effect variance as a novel sufficiency radius. With correlated Q the conditional variance
remains well defined, but it must not be interpreted through the independent ANOVA decomposition.

## Further primary-source checks for the full study (2026-10-05)
- Thorben Funke, Megha Khosla, Mandeep Rathee and Avishek Anand, *Zorro: Valid, Sparse, and Stable
  Explanations in Graph Neural Networks*, IEEE TKDE 35(8):8687–8698 (2023), DOI 10.1109/TKDE.2022.3201170.
  https://arxiv.org/html/2105.08621v2, Definition 4. Its RDT-fidelity is prediction agreement after
  jointly resampling unretained inputs under a specified noise distribution. Thus neither joint
  resampling nor conditional sufficiency is new here. Distinguish distance-resolved sign dependence,
  controlled correlated laws and agreement versus probability variance; retain original-prediction fidelity.
- Xu Zheng, Farhad Shirani, Tianchun Wang, Wei Cheng, Zhuomin Chen, Haifeng Chen, Hua Wei and
  Dongsheng Luo, *Towards Robust Fidelity for Evaluating Explainability of Graph Neural Networks*,
  ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/34293d684b1012ed45c3274b4a7edc00-Abstract-Conference.html
  Distribution shift from removing subgraphs can invalidate explanation-fidelity interpretations.
  Directly relevant to our stress-Q caveat and the separation of perturbation stability from cropping.
  We do not claim a new distribution-shift correction or a recovered observational conditional law.
- Xinyi Gao, Wentao Zhang, Junliang Yu, Yingxia Shao, Quoc Viet Hung Nguyen, Bin Cui and Hongzhi Yin,
  *Accelerating Scalable Graph Neural Network Inference with Node-Adaptive Propagation*,
  ICDE 2024, DOI 10.1109/ICDE60146.2024.00236; accepted manuscript in the author institution repository.
  https://hdl.handle.net/10072/431724. Topology-based personalised propagation depths and distillation
  already accelerate inductive inference. A degree-based computation selector or generic early exit is
  not a novelty claim here; compare actual fidelity and total cost under the stated signed-link workload.
- Andrea Giuseppe Di Francesco, Maria Sofia Bucarelli, Franco Maria Nardini, Raffaele Perego,
  Nicola Tonellotto and Fabrizio Silvestri, *Early-Exit Graph Neural Networks*, arXiv:2505.18088 (2025).
  https://arxiv.org/abs/2505.18088. Confidence-aware node/graph exits already trade propagation for cost.
  Treat as an inference-efficiency precedent; our budget check does not introduce early-exit architectures.
- Dun Ma, Jianguo Chen, Wenguo Yang, Suixiang Gao and Shengminjie Chen,
  *Pruning for GNNs: Lower Complexity with Comparable Expressiveness*, ICML 2025,
  PMLR 267:41854–41889. https://proceedings.mlr.press/v267/ma25e.html.
  Graph sparsification and computation reduction are established research topics. Any practical claim
  must concern the validated signed-network decision and include preprocessing, batching and policy costs.

The prospective contribution is a validity map and empirical mechanism study for signed dependence,
with a tested practical implication. Conditional variance, joint resampling, finite-sample bounded-mean
estimation, sufficiency and adaptive propagation each have substantial prior art.
