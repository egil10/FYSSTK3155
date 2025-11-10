# Project 2 — Neural Networks for Classification and Regression

**Course:** Data Analysis and Machine Learning (FYS-STK3155/FYS4155)  
**Institution:** University of Oslo, Norway  
**Project window:** October 14, 2025 – November 10, 2025 (deadline at midnight)  
**Group members:** Bror Johannes Tidemand Ruud (640394), Egil Furnes (693784), & Ådne Rolstad (693783).

---

## 1. Contents

1. Deliverables  
2. Project brief  
3. Methods to implement and analyse  
4. Background literature  
5. Reporting guidelines for numerical projects  
6. Electronic delivery format

---

## 2. Deliverables

Please form a group (2–3 students recommended) on Canvas before you start. When submitting Project 2 on Canvas, upload the following as a group:

- `Project2_Report.pdf`: approximately 5 000 words (≈10–12 pages) excluding references and appendices, with 10–15 figures in the main text. Additional figures/tables may go in appendices or the repository.
- A Canvas comment linking to the GitHub repository (or subfolder) that mirrors the submission. The repository must include a printable version of the report (PDF) and the complete code base.
- `Code/`: folder containing all Python modules and Jupyter notebooks needed to reproduce results. Make runs deterministic by setting seeds for random initialisation and train/test splits.
- `README.md`: this file. Include group member names, installation instructions, notebook descriptions, and usage guidance.
- Any supplemental material (figures, tables, logs) referenced in the report.

Always cite external material (lecture notes, textbooks, libraries, web resources, AI-generated code or text). If you rely on tools such as ChatGPT, include citations and, when possible, append the prompts and responses as supplemental material.

Feel free to propose alternative data sets, provided you justify your choice and compare your findings to published results. The UCI ML Repository and Kaggle are good starting points.

---

## 3. Project Brief

The goal is to implement, from scratch, a feed-forward neural network (FFNN) capable of solving both regression and multiclass classification problems. The project mirrors the week 41–42 lecture material and extends Project 1 by reusing gradient descent components.

### 3.1 Core tasks

| Part | Focus | Key requirements |
|------|-------|------------------|
| a | Analytical warm-up | Derive MSE, binary cross-entropy, and multiclass cross-entropy losses (with/without L1/L2) and the gradients of Sigmoid, ReLU, Leaky ReLU |
| b | FFNN regression | Implement FFNN with flexible architecture, initialise weights/biases, use Sigmoid hidden layers and appropriate output activation. Compare to OLS results from Project 1 on the 1D Runge function. Explore architectures (1–2 hidden layers, 50/100 neurons). |
| c | Library comparison | Benchmark against `scikit-learn`, TensorFlow/Keras, or PyTorch implementations. Validate gradients with Autograd or JAX (optional but recommended). |
| d | Activation & depth study | Compare Sigmoid, ReLU, Leaky ReLU; vary layer depth/width; discuss overfitting/bias-variance effects. |
| e | Regularisation | Add L1/L2 penalties (λ hyperparameters). Compare with Ridge and Lasso from Project 1 using the same data splits. |
| f | Classification | Adapt network for MNIST classification using Softmax cross-entropy. Report accuracy, explore scaling, hyperparameters, architectures, and compare with logistic regression or library baselines. |
| g | Critical evaluation | Summarise strengths/weaknesses of each algorithm for regression vs classification, including optimisation strategies. |

### 3.2 Suggested datasets

- Regression: 1D Runge function `f(x) = 1 / (1 + 25x²)` (baseline), optional 2D Runge or more complex surfaces.
- Classification: MNIST (full dataset via `fetch_openml('mnist_784')`), optional Fashion-MNIST or other public datasets.

Scaling inputs (e.g., dividing MNIST pixel values by 255) is recommended. Use consistent train/test splits (e.g., `train_test_split` with fixed `random_state`).

---

## 4. Methods to Implement and Analyse

### 4.1 Required implementation

1. Reuse Project 1 regression code (OLS, Ridge, Lasso) as regression benchmarks.  
2. Implement an FFNN with:
   - Configurable layers, neurons per layer, and activation per layer (Sigmoid, ReLU, Leaky ReLU, Linear, Softmax).
   - Switchable loss functions (MSE for regression, cross-entropy for classification).
   - Optional L1/L2 regularisation on weights/biases (for gradient computation).  
3. Implement backpropagation to compute gradients.  
4. Reuse and adapt gradient descent optimisers from Project 1: plain GD, SGD, SGD+Momentum, RMSprop, Adam.  
5. Integrate data scaling and train/test splitting utilities (preferably via `scikit-learn`).  
6. Track performance metrics: MSE for regression, accuracy (and optional confusion matrices) for classification.

### 4.2 Required analysis

- Discuss pros/cons of Project 1 methods vs neural networks on the provided regression task.  
- Examine hyperparameter effects (layers, neurons, activations, L1/L2) and highlight the most insightful results (heatmaps recommended).  
- Evaluate neural network strengths/limitations for regression and classification.  
- Compare optimisation strategies and learning rates.  

### 4.3 Optional extensions (choose at least two)

- Logistic regression baseline (equivalent to FFNN with no hidden layers).  
- Automatic differentiation check (Autograd/JAX) for gradients.  
- Comparison with PyTorch or similar ML frameworks.  
- Alternate datasets (Fashion-MNIST, 2D Runge, etc.).  
- Confusion matrix analysis of the best classifier.  
- Any other well-justified extension agreed upon with course staff.

---

## 5. Repository Structure

```
PROJECT 2/
├── Code/
│   ├── Implementations/
│   │   ├── neural_network.py      # FFNN definition, backpropagation, training loops
│   │   ├── activations.py         # Sigmoid, ReLU, LeakyReLU, Linear, Softmax + derivatives
│   │   ├── losses.py              # MSE, BCE, Cross-Entropy + gradients, L1/L2 penalties
│   │   ├── optimizers.py          # Plain GD, SGD, Momentum, RMSprop, Adam
│   │   ├── prepare_data.py        # Scaling, train/test splitting, polynomial features
│   │   └── plot_style.py          # Shared matplotlib styling utilities
│   ├── Code_Project1/             # Benchmarks from Project 1 (OLS, Ridge, Lasso)
│   ├── Notebooks/                 # Results and analysis notebooks (see below)
│   ├── Plots/                     # Exported figures used in the report
│   ├── Tables/                    # Generated tables/metrics
│   ├── Testing/                   # Unit tests validating each module
│   └── requirements.txt           # Python dependencies
├── project2.ipynb                 # Assignment text
└── README.md                      # This document
```

### Notebook overview

- `Notebooks/Exa.ipynb`: Analytical derivations for loss functions and activations.  
- `Notebooks/Exb.ipynb`: Baseline FFNN regression on Runge function.  
- `Notebooks/Exb_with_noise.ipynb`: Regression robustness with injected noise.  
- `Notebooks/Exc.ipynb`: Comparison with `scikit-learn`, TensorFlow/Keras, or PyTorch baselines.  
- `Notebooks/Exd.ipynb`: Activation and depth experiments, heatmaps, overfitting discussion.  
- `Notebooks/Exe.ipynb`: L1/L2 regularisation sweeps, comparison with Ridge/Lasso.  
- `Notebooks/Exf.ipynb`: MNIST classification, accuracy metrics, optional confusion matrices.  

Adapt notebook paths/imports if you restructure directories.

---

## 6. Installation and Execution

### 6.1 Prerequisites

- Python ≥ 3.8  
- `pip` (or `conda`)  
- (Optional) GPU-enabled environment for faster MNIST training

### 6.2 Setup

```powershell
cd "PROJECT 2/Code"
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Install Jupyter if needed:

```powershell
pip install jupyter
```

### 6.3 Running notebooks

```powershell
cd "PROJECT 2/Code"
jupyter notebook
```

Ensure the notebooks can import modules from `Implementations/`. Most notebooks prepend the project root to `sys.path`; adjust if you change the folder layout.

### 6.4 Running experiments via CLI

Example usage (replace with actual script entry points if provided):

```powershell
python Implementations/prepare_data.py --dataset mnist --scale
python Implementations/neural_network.py --config configs/run_regression.yaml
```

Scripts should accept seeds via command-line flags or configuration files to preserve reproducibility.

### 6.5 Running tests

```powershell
cd "PROJECT 2/Code"
python -m pytest Testing/
```

---

## 7. Results Management

- Figures stored in `Plots/` should be referenced in the report (10–15 in main text). Provide captions, axis labels, and consistent styling.  
- Tables in `Tables/` summarise hyperparameter sweeps, benchmark comparisons, and evaluation metrics.  
- Maintain a changelog or experimental log (e.g., `Logs/` or appendix) documenting key runs, learning rates, seeds, and remarks on convergence behaviour.

---

## 8. Background Literature

- Nielsen, M. **Neural Networks and Deep Learning**. http://neuralnetworksanddeeplearning.com/  
- Goodfellow, I., Bengio, Y., Courville, A. **Deep Learning** (Ch. 6–8). https://www.deeplearningbook.org/  
- Raschka, S. et al. **Machine Learning with PyTorch and Scikit-Learn** (Ch. 11–13). https://sebastianraschka.com/blog/2022/ml-pytorch-book.html  
- UiO course compendium and lecture notes Weeks 41–42: https://compphysics.github.io/MachineLearning/  
- Optimisation reference: Goodfellow et al., *Deep Learning*, Chapter 8.  
- Optional comparative study: https://medium.com/ai-in-plain-english/comparison-between-logistic-regression-and-neural-networks-in-classifying-digits-dc5e85cd93c3

---

## 9. Writing and Referencing Guidelines

- Follow scientific/technical report conventions: abstract, introduction, methods, results, discussion, conclusion, references, appendices.  
- Include pseudocode or algorithm summaries for major implementations.  
- Provide well-commented source code (in notebooks or `.py` files).  
- Validate against analytical solutions or known baselines where possible.  
- Assess numerical stability and discuss uncertainty (e.g., variance across seeds).  
- Reflect on lessons learned, potential improvements, and open questions.  
- Keep a lab log of experiments, parameter choices, and runtime observations.

When citing software or datasets, follow the recommended citation style (e.g., `scikit-learn` guidelines, Kaggle dataset citations). Cite AI tools (ChatGPT or others) if used, and retain interaction records as supplemental material when feasible.

---

## 10. Electronic Delivery Format

- Submit via Canvas: https://www.uio.no/english/services/it/education/canvas/  
- Upload only the report PDF or a link to the GitHub repository if permitted.  
- Source code lives in GitHub/GitLab (or equivalent); include instructions for reproducing key results. Avoid uploading third-party library files unless modified.  
- In the repository, maintain a folder of selected outputs (plots, tables, logs) referenced in the report.

Collaboration is encouraged; ensure all contributions are acknowledged in the report and repository.

---

## 11. Contact

For clarifications, use course discussion forums, lab sessions, or contact the teaching team. Please notify us early if you plan major deviations (alternative datasets, architectures, tooling) so we can agree on evaluation criteria.

