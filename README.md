# EUREKA

EUREKA learns an executable regression equation from observations collected in different environments, such as sites, seasons, or operating conditions. It searches sparse global and piecewise expressions and refits their coefficients across source environments to select an equation. The fitted model predicts from numeric features alone.

## Install

Use CPython 3.12 or later, up to 3.14. From the repository root, create and activate a virtual environment.

| Platform | Create the environment | Activate it |
| --- | --- | --- |
| macOS or Linux | `python3 -m venv .venv` | `source .venv/bin/activate` |
| Windows PowerShell | `py -3.12 -m venv .venv` | `..venvScriptsActivate.ps1` |

~~~sh
python -m pip install .
eureka --help
~~~

The bundled spreadsheets and DGReg cases use additional readers and a source feature screen. Install them with `python -m pip install ".[benchmarks]"` to run every dataset listed below.

## Fit and predict

~~~sh
eureka fit examples/measurements.csv --features temperature flow --target pressure --environment site --holdout site-c --out runs/example
eureka predict runs/example/model.json examples/measurements.csv --out runs/example/predicted.csv
~~~

Replace the CSV and column names with yours. Use finite numeric values for features and the target, and a label identifying each measurement environment. Use at least two source environments for leave-one-environment-out selection. `--holdout` reserves a complete environment for scoring. Omit it to use all environments as sources. EUREKA splits source rows within each environment for validation, or accepts a separate source CSV through `--validation`.

`model.json` stores the executable expression, feature order, and source-training transforms. `equation.txt` displays the equation and transforms for inspection. Use the JSON model for full-precision prediction. A held-environment run also writes `metrics.json` and `predictions.csv`. A prediction CSV needs the fitted feature columns and can omit the environment and target columns.

### Python API

~~~python
import pandas as pd
from eureka import EurekaModel, fit

frame = pd.read_csv("examples/measurements.csv")
features = ["temperature", "flow"]
source = frame[frame["site"] != "site-c"]
held = frame[frame["site"] == "site-c"]

model = fit(source[features], source["pressure"], source["site"],
            feature_names=features, target_name="pressure", seed=0)
print(model.equation())
model.save("pressure-model.json")
prediction = EurekaModel.load("pressure-model.json").predict(held[features])
~~~

Pass `validation=(X_val, y_val, env_val)` to provide your own source validation rows. Supply prediction features in the same order you used to fit.

## Run a built-in dataset

The benchmark command fits a new model and writes the same JSON and equation files. Start with the high-speed pressure holdout and apply its model to the exported measurements.

~~~sh
eureka benchmark wind --seed 0 --check --out runs/wind
eureka predict runs/wind/model.json runs/wind/predictions.csv --out runs/wind/reloaded.csv
~~~

The first command reports NRMSE about `0.808012`. `predictions.csv` contains held measurements and the newly fitted predictions. The second command adds a prediction column from the saved model.

| Dataset | Argument to `eureka benchmark` | Held conditions or splits | Reference NRMSE |
| --- | --- | ---: | ---: |
| Building heating | `heating` | 4 | 0.403 |
| Concrete strength | `concrete` | 4 | 0.379 |
| Daily bicycle rentals | `bicycle` | 8 | 0.735 |
| High-speed pressure | `wind` | 1 | 0.808 |
| Steel energy | `steel` | 4 | 0.00525 |
| Synthetic shortcuts | `shortcut-a`, `shortcut-b` | 50 each | 0.069, 0.069 |
| DGReg nonlinear and periodic | `dgreg-a`, `dgreg-b` | 20 each | 0.055, 0.053 |

List unit names and run one split before starting a whole dataset.

~~~sh
eureka benchmark shortcut-a --list-units
eureka benchmark shortcut-a --unit "A|2026110201" --seed 0 --check --out runs/shortcut
eureka benchmark heating --jobs 4 --out runs/heating
~~~

Leave out `--unit` and `--seed` to use all conditions and fitting seeds for the selected dataset. The larger runs can take several minutes or longer on a CPU. Each fit writes a model and predictions. `scores.csv` records the measured errors and supplied reference values. `summary.json` averages the median NRMSE per condition or split. NRMSE divides held RMSE by the population standard deviation of the source-training response. `--check` returns an error if any selected fit falls outside the numeric reference tolerance.

## Prepare the original wind data

The built-in wind case uses compact arrays prepared from the Causal Chambers wind tunnel. To rebuild the arrays from the original archive, run the commands below. The preparer downloads the ZIP, checks its SHA-256, calibrates and aggregates the measurements, and partitions the speed ranges. Supply `--archive path/to/wt_bernoulli_v1.zip` to use a local copy.

~~~sh
eureka prepare-wind --out downloads/wind
eureka benchmark wind --prepared-wind downloads/wind --check --out runs/wind-from-raw
~~~

See [data sources and transformations](DATA.md) for the upstream download and the input definitions.

## License

EUREKA's source code is licensed under the [MIT License](LICENSE). See [DATA.md](DATA.md) for the licenses and attribution of bundled datasets.
