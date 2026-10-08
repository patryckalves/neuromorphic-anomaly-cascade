# Statement of AI Assistance & Research Integrity

In compliance with the **Committee on Publication Ethics (COPE)** position on artificial intelligence tools, the **STM Association Global Reporting Standard for AI Disclosure in Research**, and the **Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq) Diretrizes de Integridade Científica (Portaria CNPq nº 2.664/2026)**, this document provides a full, transparent disclosure of the scope, nature, and verification protocol of AI-assisted tools utilized in the development, benchmarking, and documentation of this repository.

---

## 1. Tool Specification

- **Platform & Environment**: Google DeepMind Antigravity Agentic Pair-Programming Environment.
- **Model Foundations**: Gemini 2.8 Flash and Claude 5.5 Sonnet.
- **Access Period**: April 2026.
- **Operating Context**: Local, sandboxed development environment running on Apple Silicon (macOS).

---

## 2. Structured Disclosure of Contributions (CRediT-Adapted Taxonomy)

Following the taxonomy proposed by the STM / COPE / WCRIF working group on AI disclosure in research:

| Research & Development Domain | AI Contribution Scope | Human Researcher Oversight & Responsibility |
| :--- | :--- | :--- |
| **Conceptualization & Mathematical Hypotheses** | Served as an interactive sounding board; assisted in exploring biophysical formulations (Tsodyks-Markram synaptic depression, leaky reservoir spillover dynamics, sigmoidal dendritic gating). | Selected hypotheses, derived continuous ODE formulations, verified dimensional consistency, and made all final architectural decisions. |
| **Code Implementation & Refactoring** | Generated draft implementations of vectorized NumPy routines, boilerplate class wrappers, and runner scripts. | Audited all source code line-by-line; validated algorithmic time complexity ($O(1)$ per timestep), memory bounds, and numerical stability (preventing division-by-zero and exponent overflows). |
| **Benchmark Automation & Execution** | Automated shell orchestration (`scripts/run_wave9_batch.sh`, parallel sub-process spawning) to run the 58 datasets across Numenta Anomaly Benchmark (NAB). | Designed test protocols, supervised all execution runs, verified threshold optimization criteria, and confirmed raw output CSVs against official scoring metrics. |
| **Data Analysis & Visualization** | Parsed score summaries, computed subcategory aggregations, and assisted in formatting comparative performance tables. | Checked all scores for mathematical consistency, calculated precision-recall trade-offs, and verified statistical significance against official baseline tables. |
| **Technical Writing & Documentation** | Provided initial structural formatting and English grammar reviews for repository documentation and code docstrings. | Authored all core scientific arguments, ensured strict epistemic modesty, articulated model limitations, and established the overall tone. |

---

## 3. Human Authorship & Non-Delegable Accountability

1. **Non-Authorship of Non-Human Agents**: In accordance with COPE and CNPq guidelines, no artificial intelligence system is designated as an author or co-author. Authorship requires legal personhood, moral agency, and the capacity to bear public accountability for scientific claims—criteria that computational models cannot meet.
2. **Exclusivity of Human Responsibility**: The human author holds sole and undivided responsibility for the validity, empirical rigor, and scientific claims published within this repository. Any potential bugs, numerical discrepancies, or interpretive errors are solely the responsibility of the human researcher.
3. **Empirical Verifiability & Falsifiability**: No benchmark scores, metrics, or experimental results reported in this codebase are synthetically hallucinated or extrapolated. Every metric reported is deterministically reproducible using the included benchmark scripts running against the official NAB dataset repository.
4. **Adherence to Institutional Standards**: This disclosure satisfies the mandatory disclosure rules outlined in **Portaria CNPq nº 2.664 de 6 de março de 2026** regarding the ethical, transparent, and responsible use of generative artificial intelligence in Brazilian and international scientific activity.
