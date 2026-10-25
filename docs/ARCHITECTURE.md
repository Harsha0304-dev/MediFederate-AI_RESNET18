# Architecture

## X-ray
X-ray -> preprocessing -> pretrained ResNet18 -> NORMAL/PNEUMONIA.

## Federated learning
S3/institution data -> simulated hospital partitions -> local ResNet training -> optional clipping/noise -> optional 8-bit quantization -> FedAvg -> global model.

## CT
CT volume -> segmentation model (U-Net/MONAI-ready) -> binary abnormal-region mask -> measurements using pixel/voxel spacing -> optional OpenAI research explanation.

## Cloud
AWS S3 stores controlled copies of datasets and project artifacts. S3 credentials are never hard-coded. Use IAM with least privilege.
