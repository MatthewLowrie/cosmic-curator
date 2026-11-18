# Cosmic Curator: Exoplanet Signal Lab

An enterprise-ready, production-grade machine learning pipeline that transforms raw astronomical observations from the NASA Exoplanet Archive into an interactive web dashboard.

This repository demonstrates advanced software architecture patterns, feature auditing, target leakage exposure, and high-performance pipeline orchestration in Python.

---

## 🌌 Architectural Philosophy

This codebase was graduated from a monolithic data science script into a highly maintainable, modular package adhering to modern engineering standards:

1. **Separation of Concerns (SoC):** Every file has a single, strict responsibility. Data engineering, modeling, charting, compiling, and configuring are completely isolated.
2. **Single Responsibility Principle (SRP):** Functions are small, highly focused, and pass immutable parameters or strictly defined DataFrames.
3. **Robustness & Typings:** Every function is fully type-hinted, enabling early-stage static analysis and excellent IDE autocompletion in VS Code.
4. **Structured Configuration:** Loose constants have been consolidated into a frozen, read-only Python dataclass to prevent runtime mutations.
5. **Centralized Logging:** Raw `print()` statements are replaced with standard Python logging, appending clean, timestamped UTC tracking outputs.

---

## 📁 Repository Structure

The project implements the standard **src layout** pattern:

```text
cosmic_curator/
│
├── src/                          # Core application package folder
│   ├── __init__.py               # Package marker file (enables absolute imports)
│   ├── config.py                 # Hyperparameters, feature schemas, and logging
│   ├── data_loader.py            # API client for the NASA Exoplanet Archive
│   ├── model_engine.py           # ML modeling, feature auditing, and metrics
│   ├── visualizer.py             # Plotly graph configurations
│   └── compiler.py               # HTML serializations and pandas table compilation
│
├── docs/                         # Automatically created directory
│   └── index.html                # Final, fully assembled interactive dashboard
│
├── template.html                 # Pure HTML skeleton with placeholder variables
├── main.py                       # Root application coordinator (entry point)
└── README.md                     # Documentation (this file)
```

---

## 🛠️ The Assembly Modules

### 1. `main.py`

The orchestrator. It manages no raw technical logic; instead, it catches runtime exceptions, manages a graceful `try...except` wrapper, and coordinates the sequential pipeline phases.

### 2. `src/config.py`

Holds the `PipelineConfig` dataclass. It controls our seed variables, TAP coordinates, physical and leaky feature collections, color palettes, and file paths.

### 3. `src/data_loader.py`

Constructs web-safe Astronomical Data Query Language (ADQL) strings and executes a stream query to NASA's Table Access Protocol (TAP) endpoint.

### 4. `src/model_engine.py`

The computational core. It encapsulates imputation, scaling, and classification into Scikit-learn `Pipeline` structures. It handles:

- **The Leaky Baseline:** Deliberately includes human-curated administrative flags to showcase target leakage (~0.999 ROC-AUC).
- **The Feature Audit:** Retrains models strictly on physical exoplanet dimensions to create an honest scientific classifier.
- **Metric Outputs:** Generates ROC-AUC coordinates, confusion matrices, and impurity-based feature importances.

### 5. `src/visualizer.py`

Configures five complex, interactive Plotly visualizations matching dark-space themes:

- Kepler Candidate Population (Scatter Map)
- Model Performance Audit (Bar Chart)
- Feature Importances (Horizontal Bar Chart)
- Audited Confusion Matrix (Heatmap)
- Unresolved Candidates prediction probability (Scatter Map)

### 6. `src/compiler.py`

Dumps Plotly elements to lightweight, standalone HTML wrappers. It isolates pandas structures, parses them to high-density HTML tables, loads the `template.html` layout, and compiles the final dashboard.

---

## 🚀 Getting Started (Windows)

### Prerequisites

Ensure you are running Python 3.10+ and have your virtual environment activated:

```powershell
# Activate your virtual environment in PowerShell
.venv\Scripts\Activate.ps1
```

### Execution

To pull live data from NASA's databases, train the models, compile charts, and build your dashboard, run the orchestrator:

```powershell
python main.py
```

### Viewing the Output

To launch your finished interactive dashboard directly in your default browser, run:

```powershell
Start-Process "docs/index.html"
```

---

## 📊 Scientific Insights

This pipeline visualizes a critical lesson in responsible AI: **Target Leakage**.

A model trained with NASA's administrative features (like `koi_score`) produces a suspiciously perfect test score of **0.999**. However, this model is useless in a real-world observatory because those administrative flags are calculated _after_ the planet is confirmed.

By running the feature audit and training solely on physical measurements (like radius, transit depth, and orbital period), we achieve a scientifically honest classifier that generalizes beautifully to unresolved signals, surfacing the top candidate priorities for further astronomical observation.
