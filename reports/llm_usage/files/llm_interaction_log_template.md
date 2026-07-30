# LLM Interaction Log

This file records all relevant interactions with Large Language Models (LLMs) used during the project.

Each interaction must be recorded as a separate entry. You must use the following unique interaction ID: `G03-06-12-001`, `G03-06-12-002`, etc., where GXX is the group identifier, MM-DD means month and day, followed by a progressive number starting from 001 for the first recording of the day.

Do **not** overwrite previous entries. Do **not** use the same downloaded file name twice. All downloaded files must be stored in the project repository and referenced exactly with their relative path.

The same day must not be covered by more than one interaction log. If you feel that the interaction log is becoming too long, at the end of the day close the log and start another one the following day. The logs are identified by their log_id, with the group code and a progressive number NN: "GXX-NN". Each log must be contained in a file named: "LOG-GXX-NN.md". 

---

## Metadata

```yaml
group_id: "G32"
repository_url: "https://github.com/simmassimo/isw-qubo-classification-G32.git"
students:
  - matricola: "60/61/65985"
    name: "Simone Massenti"
  - matricola: "60/61/65904"
    name: "Alexandra Brelstaff"
last_update: "YYYY-MM-DD"
log_id: "G32-NN"
```

---

# Interaction Entries

---

## Interaction GXX-MM-DD-001

### 1. LLM and chat information

```yaml
llm_name: "Copilot"
llm_version_or_model: "e.g. GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro"
chat_name_or_identifier: "Short name or identifier of the active chat"
interaction_mode: "web_chat"
```

### 2. Author of the interaction

```yaml
performed_by: "65904"   # use "couple" or the matricola of the individual student
```

### 3. Project phase

Select one or more phases.

```yaml
project_phase:
  - "requirements_understanding"
  - "preprocessing"
  - "feature_selection_qubo"
  - "optimization_algorithm"
  - "classification_model"
  - "testing"
  - "debugging"
  - "documentation"
  - "gui"
  - "repository_structure"
  - "other"
```

### 4. Input files and/or context provided to the LLM

List any files or project documents provided to the LLM.
Record ("yes"/"no") if you provided code snippets, error messages, dataset excerpts.
If the prompt was itself generated or improved by another LLM interaction, mention the source interaction ID.

```yaml
input_context:
  files_uploaded:
    - file_name: null
      repository_path: null
      description: null
  code_snippets: null
  error_messages: null
  dataset_excerpts: null
  prompt_generated_by: null 
```

### 5. Student prompt

Paste the exact prompt sent to the LLM. 

```text
For a dense dataset, when you apply QUBO for feature reduction what are the recommended binary classification algorithms?
```

### 6. LLM response

Paste the full response received from the LLM. If the response is very long, it is acceptable to store it in a separate file and reference it here.

```text
### Recommended binary classifiers after QUBO feature reduction for dense datasets

Below is a concise comparison table of **practical, well‑performing binary classifiers** to try after you reduce features with a QUBO-based method on a dense dataset. The table highlights the attributes most relevant when you’ve already done feature selection (stability of features, reduced dimensionality, possible correlations).

| **Classifier** | **Why it fits after QUBO** | **When to prefer** |
|---|---:|---|
| **Logistic Regression (L2 / regularized)** | Simple, fast, interpretable; benefits from fewer, informative features; regularization handles residual multicollinearity. | Baseline model; when interpretability and calibration matter. |
| **Support Vector Machine (linear or RBF)** | Strong with medium-dimensional, well-separated features; linear SVM scales well after reduction; RBF handles nonlinearity if needed. | When margin maximization and robustness to outliers are desired. |
| **Random Forest** | Nonlinear, robust to remaining noisy features; handles feature interactions without heavy tuning. | When you want strong off-the-shelf performance and feature‑importance diagnostics. |
| **Gradient Boosting (XGBoost / LightGBM / CatBoost)** | High predictive power on tabular data; handles complex interactions and residual noise after selection. | When maximizing accuracy is primary and you can tune hyperparameters. |
| **Regularized Linear Models (Elastic Net)** | Combines L1/L2 to keep sparsity and stability; complements QUBO’s selection by further shrinking weak features. | When some correlated features remain and you want sparse, stable coefficients. |
| **k-Nearest Neighbors (k-NN)** | Simple, nonparametric; can work well if QUBO yields compact, discriminative feature sets and dataset size is moderate. | Small-to-medium datasets with meaningful distance metrics. |
| **Naive Bayes (Gaussian / Bernoulli)** | Extremely fast; effective if selected features are approximately conditionally independent. | Very large datasets or strict latency constraints; as a quick baseline. |
| **Neural Network (small MLP)** | Flexible nonlinear modeling for complex decision boundaries; benefits from fewer inputs after QUBO. | When you expect complex patterns and have enough data to avoid overfitting. |

---

### Practical guidance and ordering
1. **Start simple**: Fit **Logistic Regression (with L2)** and a **linear SVM** as baselines — they’re fast and reveal whether the reduced features are linearly separable.  
2. **If linear models underperform**, try **tree ensembles** (Random Forest, then Gradient Boosting). These often give the largest accuracy gains on tabular data.  
3. **If you need sparsity or interpretability**, use **Elastic Net** or retrain Logistic Regression with L1 to force sparser coefficients.  
4. **If nonlinearity remains**, test **RBF SVM** or a **small MLP**. Use cross‑validation to avoid overfitting.  
5. **Use Naive Bayes or k‑NN** only as quick baselines or when their assumptions match your reduced feature set.  

---

### Tuning and validation tips after QUBO selection
- **Cross‑validate end‑to‑end**: include the QUBO selection step inside cross‑validation (or use nested CV) to avoid optimistic bias.  
- **Check stability**: run QUBO selection multiple times (or with bootstraps) and verify classifier performance is stable across feature subsets.  
- **Regularize aggressively**: even after reduction, regularization (L1/L2, tree depth, learning rate) prevents overfitting.  
- **Calibrate probabilities**: if you need well‑calibrated outputs, apply Platt scaling or isotonic regression after training.  
- **Inspect feature importances**: for tree models, compare importances to QUBO-selected features to validate selection quality.

---

### Quick recommendation (if you want a single starting point)
- **Logistic Regression (L2)** as baseline → **Gradient Boosting (LightGBM/XGBoost)** for best accuracy if baseline is insufficient.

If you want, I can:  
- suggest hyperparameter grids for any of these models,  
- show how to embed the QUBO selection inside nested cross‑validation, or  
- recommend a short experiment plan (3–5 models, CV setup, metrics) tailored to your dataset size and class balance.
```

If the full response is stored in a separate file:

```yaml
response_stored_in_file:
  used: false
  file_name: null
  repository_path: null
```

### 7. Files generated or downloaded from the LLM response

List every file produced by the LLM and downloaded or copied into the project. Each file name must be unique across the whole repository.

```yaml
generated_or_downloaded_files:
  - unique_file_name: null
    repository_path: null
    file_type: null       # e.g. py, ipynb, md, yaml, csv, json, txt
    created_from_response: true
    short_description: null
    referenced_in_response: true
```

### 8. How the LLM output was used

```yaml
usage_of_output:
  used_without_changes: null  # true / false
  modified_before_use: null   # true / false
  description_of_modifications: null
  related_repository_files:
    - null
```

### 9. Problems, errors, or hallucinations

```yaml
issues_found:
  any_issue_found: null    # "yes"/"no"  The subsequent part only if "yes"
  issue_categories:    # Delete non-pertinent ones
    - "none"
  description: null
  how_issue_was_resolved: null
```

### 10. Usefulness and reliability assessment

Use a scale from 1 to 5 (1 = lowest grade, 5 = highest grade).

```yaml
assessment:
  usefulness_1_to_5: 3
  correctness_1_to_5: 4
  clarity_1_to_5: 2
  confidence_after_verification_1_to_5: 2
  would_reuse_this_output: null
  notes: null
```

---

## Interaction GXX-MM-DD-001

### 1. LLM and chat information

```yaml
llm_name: "Copilot"
llm_version_or_model: "e.g. GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro"
chat_name_or_identifier: "Short name or identifier of the active chat"
interaction_mode: "web_chat / IDE_assistant / API / desktop_app / other"
```

### 2. Author of the interaction

```yaml
performed_by: "couple"   # use "couple" or the matricola of the individual student
```

### 3. Project phase

Select one or more phases.

```yaml
project_phase:
  - "requirements_understanding"
  - "preprocessing"
  - "feature_selection_qubo"
  - "optimization_algorithm"
  - "classification_model"
  - "testing"
  - "debugging"
  - "documentation"
  - "gui"
  - "repository_structure"
  - "other"
```

### 4. Input files and/or context provided to the LLM

List any files or project documents provided to the LLM.
Record ("yes"/"no") if you provided code snippets, error messages, dataset excerpts.
If the prompt was itself generated or improved by another LLM interaction, mention the source interaction ID.

```yaml
input_context:
  files_uploaded:
    - file_name: null
      repository_path: null
      description: null
  code_snippets: null
  error_messages: null
  dataset_excerpts: null
  prompt_generated_by: null 
```

### 5. Student prompt

Paste the exact prompt sent to the LLM. 

```text
In numpy how do you update a single row of a 2d array?
```

### 6. LLM response

Paste the full response received from the LLM. If the response is very long, it is acceptable to store it in a separate file and reference it here.

```text
### Update single row in NumPy

**Quick answer:** index the row and assign a new 1D array or sequence.  

import numpy as np
a = np.arange(12).reshape(3,4)   # shape (3,4)
a[1] = [100, 101, 102, 103]      # replace row 1 in place


---

### Examples

#### Replace entire row

a = np.zeros((4,5), dtype=int)
a[2] = np.arange(5)              # sets row index 2 to [0,1,2,3,4]


#### Replace row slice or part of a row

a = np.arange(12).reshape(3,4)
a[0, 1:3] = [99, 98]             # updates columns 1 and 2 of row 0


#### Using broadcasting

a = np.zeros((3,4))
a[1] = 5                         # sets entire row 1 to 5 (broadcast scalar)


#### With boolean mask for row selection

mask = np.array([False, True, False])
a[mask] = np.array([[7,7,7,7]])  # mask selects row 1 and assigns a 2D row


---

### Notes and gotchas

- **Shape must match:** the assigned value must have a compatible shape (e.g., a 1D length equal to number of columns, or a scalar for broadcasting).  
- **Dtype conversion:** assignment will cast values to the array’s dtype.  
- **Fancy indexing:** using a list/array of row indices (e.g., `a[[1]] = ...`) works but can behave differently when assigning multiple rows; single-row simple indexing (`a[i]`) is the usual in-place update.  
- **In-place update:** `a[i] = ...` modifies the original array without creating a new array.

```

If the full response is stored in a separate file:

```yaml
response_stored_in_file:
  used: false
  file_name: null
  repository_path: null
```

### 7. Files generated or downloaded from the LLM response

List every file produced by the LLM and downloaded or copied into the project. Each file name must be unique across the whole repository.

```yaml
generated_or_downloaded_files:
  - unique_file_name: null
    repository_path: null
    file_type: null       # e.g. py, ipynb, md, yaml, csv, json, txt
    created_from_response: true
    short_description: null
    referenced_in_response: true
```

### 8. How the LLM output was used

```yaml
usage_of_output:
  used_without_changes: null  # true / false
  modified_before_use: null   # true / false
  description_of_modifications: null
  related_repository_files:
    - null
```

### 9. Problems, errors, or hallucinations

```yaml
issues_found:
  any_issue_found: null    # "yes"/"no"  The subsequent part only if "yes"
  issue_categories:    # Delete non-pertinent ones
    - "none"
  description: null
  how_issue_was_resolved: null
```

### 10. Usefulness and reliability assessment

Use a scale from 1 to 5 (1 = lowest grade, 5 = highest grade).

```yaml
assessment:
  usefulness_1_to_5: 5
  correctness_1_to_5: 4
  clarity_1_to_5: 3
  confidence_after_verification_1_to_5: 4
  would_reuse_this_output: null
  notes: null
```

---

# Suggested file naming convention

Use unique and descriptive names for every file generated or downloaded from an LLM response.

Recommended format:

```text
LLM_<interaction_id>_<short_description>.<extension>
```

Examples:

```text
LLM_G03-06-05-002_preprocessing_function.py
LLM_G03-06-12-001_qubo_cost_function.py
LLM_G03-06-12-001_test_feature_selection.py
LLM_G03-06-13-003_readme_section.md
```

---

# Final checklist

Before submission, verify that:

- [ ] Every relevant LLM interaction has been recorded.
- [ ] Each interaction has a unique ID.
- [ ] Date and time are present for every interaction.
- [ ] The author is identified as either `couple` or one student matricola.
- [ ] The full prompt is included.
- [ ] The full LLM response is included or correctly referenced as a separate file.
- [ ] Every downloaded/generated file has a unique name, is present in the repository and is correctly referenced in this log.
- [ ] Problems, errors, and hallucinations have been reported honestly.
