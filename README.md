# MediFederate AI — Final Upgraded Project

## Scope
1. **Chest X-ray classification:** NORMAL vs PNEUMONIA using pretrained ResNet18, augmentation, class-weighted loss, AdamW, learning-rate scheduling and validation-based early stopping/checkpointing.
2. **Federated Learning:** 3 simulated healthcare institutions, weighted FedAvg, IID and Dirichlet Non-IID experiments.
3. **Privacy mechanism:** client-update clipping + Gaussian noise. This is an implemented privacy mechanism, not a formal epsilon-DP guarantee until a validated accountant is added.
4. **Communication efficiency:** 8-bit quantization with measured payload bytes and compression reduction.
5. **CT extension:** lung nodule/lesion segmentation measurement pipeline for location, size, dimensions, area, shape, HU statistics and 3-D volume when correct spacing and a 3-D mask are available.
6. **AWS S3:** private cloud storage for controlled datasets/models/results. Current project bucket can be configured with `S3_BUCKET` and `AWS_REGION`.
7. **Streamlit:** X-ray inference, CT measurement, federated workflow and cloud/privacy pages.

## Important scientific rule
Do not reuse the old SimpleCNN results as ResNet18 results. Fresh ResNet18 experiments must be run and recorded. Higher accuracy is an experimental target, not a guaranteed number. Test data is used only for final evaluation.

## Structure
```text
MediFederate_AI_Final_ResNet18/
├── app.py
├── centralized_baseline.py
├── evaluate_centralized.py
├── run_federated_simulation.py
├── manual_federated_simulation.py
├── full_experiment.py
├── fl_client.py
├── config.py
├── model.py
├── data_utils.py
├── metrics_utils.py
├── privacy_utils.py
├── compression_utils.py
├── federated_core.py
├── requirements.txt
├── .env.example
├── configs/default.yaml
├── src/medifederate/
│   ├── model.py
│   ├── xray.py
│   ├── federated.py
│   ├── ct.py
│   ├── cloud.py
│   └── openai_report.py
├── scripts/
│   ├── train_xray.py
│   ├── evaluate_xray.py
│   ├── run_federated.py
│   ├── train_ct.py
│   ├── s3_upload.py
│   └── s3_download.py
├── app/streamlit_app.py
├── data/processed/xray/
├── data/ct/images/
├── data/ct/masks/
├── models/
├── results/
├── tests/
└── docs/
```

## Your current AWS layout
```text
s3://medifederate-ai-022741688198/
├── datasets/xray/train/
├── datasets/xray/validation/
├── datasets/xray/test/
├── datasets/ct/images/
├── datasets/ct/masks/
├── models/
└── results/
```
The X-ray dataset has already been uploaded to the private bucket. Do not upload a CT dataset until the guide confirms the exact dataset/format.

## Setup
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```
Set the S3 values in `.env`. Do not put AWS secret keys in `.env` if you are using `aws configure`; the AWS CLI credential store is preferable for local development.

## X-ray paths
```text
data/processed/xray/train/
data/processed/xray/validation/
data/processed/xray/test/
```
To use your existing dataset, copy/sync only `train`, `val`→`validation`, and `test` into those paths. The duplicate `chest_xray` and `__MACOSX` directories are not required.

## Train/evaluate ResNet18
```powershell
python centralized_baseline.py
python evaluate_centralized.py
```
The upgraded training uses augmentation, class weighting, LR scheduling and early stopping. Accuracy must be measured after training; no 90%+ result is assumed.

## Federated experiments
```powershell
python run_federated_simulation.py --rounds 5 --clients 3 --iid
python run_federated_simulation.py --rounds 5 --clients 3 --non-iid
python run_federated_simulation.py --rounds 5 --clients 3 --iid --dp-noise 0.1 --label federated_dp_01
python run_federated_simulation.py --rounds 5 --clients 3 --iid --compression --label federated_compression
python run_federated_simulation.py --rounds 5 --clients 3 --iid --dp-noise 0.1 --compression --label federated_dp_compression
```

## CT pipeline
```text
CT volume → preprocessing → trained segmentation/detection model → binary nodule mask
         → location + maximum dimension + width/height + area + shape
         → HU statistics
         → 3-D volume when a 3-D mask and voxel spacing are available
```
The included measurement code does not fake a trained CT detector. It requires a segmentation mask until the CT training pipeline is trained and validated on the selected annotated dataset.

## Streamlit
```powershell
streamlit run app/streamlit_app.py
```

## AWS
The S3 integration is in `src/medifederate/cloud.py` and the upload/download helpers are in `scripts/`. It is configured for the current region/bucket through environment variables. S3 is storage; it is not itself the federated server.

## Tests
```powershell
pytest -q
```
