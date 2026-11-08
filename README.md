# CarbonGlobe — GuGuGaGa

Code by Yangbojun Yang for the IEEE BigData Cup 2026 AI Emulation Challenge.
This is a local release intended for a private GitHub repository until the competition ends.

## Environment
Python 3.14, CPU. Install the exact versions in `requirements.txt`.
Versions were supplied by the project; no dependencies are bundled.

## Data and private artifacts
Obtain the competition data from https://www.kaggle.com/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/data
and historical CarbonGlobe from https://www.kaggle.com/datasets/zhihaow/carbonglobe .
Place SSP files and sample_submission in `data/kaggle/`, historical arrays in
`data/dataset/data_global/`, statistics in `data/dataset/data_stats/`.
Competition data, derived arrays, submission CSVs, weights and private site manifests
are excluded from this repository. The code's MIT license does not relicense data.
Provide the original `site_splits.npz` privately under `data/inputs/`;
its regeneration from public split files is not implemented in this release.
Other experiments also require original protected-site manifests and cached inputs
under `data/project/results/`, preserving their historical relative paths.

## S1
Copy config.example.json to config.json and edit its paths. Paths in the training
config resolve relative to the configuration file.
```
python src/s1/prepare_inputs.py --config config.json
python src/s1/train_s1.py --config config.json
python src/s1/verify.py --config config.json
```
Expected S1 SHA256: `23216519297719e207970ecd37151181b57cb53d1e23a11499ca33e7b0824aba`.
The archived verifier checks F2 as well and does not fail on a checksum mismatch;
compare the actual checksum separately. Do not treat that message as exact replay proof.

## Other results
Run archived entrypoints using `python run_experiment.py <script> --workspace <directory>`.
The workspace contains `data/`; this parameter replaces fixed machine paths.
F2: `src/blend/blend_f2.py --config <absolute config>`.
F3: `src/blend/score_blend_f3_20261102/run_f3.py`.
P1/P2: `src/blend/segment_probe_20261102/generate_probes.py`.
G1/G2: `src/full_sites/run_full_sites.py`.
Experiments: run the corresponding `run_*.py` or `generate.py` in `src/experiments/`.
Entry points retain historical artifact dependencies; archived report writers may
refer to original script-layout paths not shipped here. They have not been run.
Expected submitted CSV hashes/IDs/scores are in `checksums/submissions.json`.
Experiment-only results have no promised CSV hash. S1 was rerun from official
arrays with the original private site manifest and matched its expected checksum.
Other experiments were not rerun during packaging; their archived measured times
are listed below and are not promises for another machine.
F2, F3 and P1 require earlier submission CSVs and their public scores. Those CSVs
are absent; these blends cannot be reconstructed from S1 alone.

## Disclosures
The 49,204-site pool includes original benchmark test sites. Organizers have not
confirmed that use; see technical report Section 2.1. The 1,024 fresh sites were
loaded once by an interrupted run, without derived evaluation results.
F2/F3 blend weights are fitted to public leaderboard scores (Section 4.2),
which cover about 30% of test rows and may not generalize to the private part.
No final submission selection is changed by this package.

## Credits
Ideas, not copied code, were taken from Lucas Bruzzone's Kaggle notebook
“CarbonGlobe: from 0.142 to 0.060” (Apache 2.0) and Francisco Angulo de Lafuente's
“carbonglobe-emulator-2026” repository (MIT). See the draft report references.
C3 and the V6 third-batch source code are included in this snapshot.

## C3 and V6 third batch

```bash
python run_experiment.py src/experiments/c3_collapse_mixture_20261107/run_c3_experiment.py --workspace .
python run_experiment.py src/experiments/v6_c3_ssp_20261108/run_v6_c3_ssp.py --workspace .
```

These scripts need the same historical protected-site manifests and cached inputs
under `data/project/results/`; the manifests and arrays are not redistributed.
V6-6/V6-7 expected CSV hashes and submission records are in `checksums/submissions.json`.

Measured times from the saved result reports (hardware and settings of those runs;
not promises for another machine):

| Experiment | Reported seconds | Approximate hours | Source report |
|---|---:|---:|---|
| C3 | 29252.58 | 8.13 | C3 results, 2026-11-07 |
| V6 third batch | 6477.40 | 1.80 | SSP prediction results, 2026-11-08 |
| H6 | 49112.11 | 13.64 | Window stationarity results, 2026-11-04 |
| Step 2 | 19034.38 | 5.29 | Collapse predictability results, 2026-11-05 |
| Other experiments | Unrecorded here | Unrecorded here | No runtime asserted |

The source reports give C3 8.13 hours and H6 13.64 hours; these differ from
the rough estimates in the release task brief. The recorded values take precedence.

## Additional experiments and measured runtimes

C3: `python run_experiment.py src/experiments/c3_collapse_mixture_20261107/run_c3_experiment.py --workspace .`
V6 third batch: `python run_experiment.py src/experiments/v6_c3_ssp_20261108/run_v6_c3_ssp.py --workspace .`
S1 measured local preparation/training seconds: 8.707424640655518 / 162.6924729347229.
Other exact runtimes are recorded in the technical report; unavailable values remain unrecorded.
