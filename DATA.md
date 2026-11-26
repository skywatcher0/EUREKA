# Data sources

The UCI ZIPs and gzip-compressed DGReg inputs are installed with EUREKA. Input splitting and feature preparation run locally when you select a dataset. The wind example includes compact derived arrays and a command to rebuild them.

## UCI measurements

The following datasets are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included ZIPs use the upstream archive bytes. The data loaders make the transformations described here.

| Dataset | Attribution and source | Benchmark transformation |
| --- | --- | --- |
| Energy Efficiency | Athanasios Tsanas and Angeliki Xifara, [UCI 242](https://doi.org/10.24432/C51307) | Use the eight building variables and heating load. Hold out groups of three adjacent relative-compactness values. |
| Concrete Compressive Strength | I-Cheng Yeh, [UCI 165](https://doi.org/10.24432/C5PK67) | Use mixture features and curing age to predict strength. Hold out one of the age bands bounded by 7, 28, and 90 days. |
| Bike Sharing | Hadi Fanaee-T and João Gama, [UCI 275](https://doi.org/10.24432/C5W894) | Use daily weather records and square-root rental counts. Hold out a year–season pair. Align source weather rows to the dates in the daily file. |
| Steel Industry Energy Consumption | Sathishkumar V E, Changsun Shin, and Yongyun Cho, [UCI 851](https://doi.org/10.24432/C52G8C) | Use reactive-energy readings, power factors, and seconds from midnight. Keep up to 3,000 rows per quarter and hold out a quarter. |

The predictors and responses are centered and scaled from source-training rows. Predictions are converted back to the benchmark response units. For bicycle rentals the response unit is the square root of daily rentals.

## DGReg and synthetic shortcuts

DGReg `nonlin.csv.gz` and `sin.csv.gz` contain the original upstream CSV bytes from Jonathan Demke and contributors. The [DGReg dataset inventory](https://github.com/demkejon001/dgreg_bench#dataset-information) lists the nonlinear dataset under MIT and the sine dataset under CC0 1.0. The upstream MIT notice is included alongside the data. Download the original datasets from the [DGReg collection](https://dataverse.harvard.edu/dataverse/dgreg_bench).

| Original file | SHA-256 of uncompressed bytes |
| --- | --- |
| `nonlin.csv` | `8db92c0da3a32311469f0b499e9e78a3da8efbc68fd0425e50b31c40c7806933` |
| `sin.csv` | `5d5b921e4171452445d476dc7be2a2d1bd5732cc9d5b12bd36f8da623aa270b8` |

DGReg holds out domain 0 for the nonlinear task and domains 1, 2, 4, 7, 8, 11, 12, and 14 for the periodic task. The splitter caps each partition at 6,000 rows. Feature screening uses the source-training rows.

The shortcut suites generate five mechanisms from fixed seeds. Each mechanism generates 1,800 rows per environment, with three source and three held environments. One input reverses its correlation with the response mechanism across these groups. Five further inputs contain independent noise.

## Wind pressure

The high-speed example derives from `wt_bernoulli_v1` by Juan L. Gamella, Jonas Peters, and Peter Bühlmann. The measurements are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Dataset definition](https://github.com/juangamella/causal-chamber/tree/main/datasets/wt_bernoulli_v1)

The raw data occupy about 121 MB after decompression. Use `eureka prepare-wind` to read the official archive and write the fitted-input arrays. [Official ZIP download](https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/wt_bernoulli_v1.zip)

- SHA-256 `e5d58a67f6307b9099177425e91c449b9e62e7c712f8c6f34d1dc80ce4db01e3`.
- Compute a sensor offset from `fans_off.csv` after the first 2,000 readings. Average intake speed and pressure by the acquisition flag. Add the offset to the upstream-minus-downstream pressure difference.
- Normalize intake speed by 3,000 rpm. Reserve normalized speeds of 0.80 and above for testing. Divide source observations into speed bands of width 0.20 and split each band into training and validation rows. Scale pressure using source-training observations.
