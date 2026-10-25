# Software Requirements Specification

## Application
MediFederate AI: privacy-preserving medical image analysis.

## Features
- ResNet18/transfer learning chest X-ray classification.
- Multi-hospital Federated Learning simulation.
- FedAvg.
- IID and Dirichlet Non-IID.
- Differential-privacy noise mechanism with clipping.
- 8-bit update compression and communication measurement.
- CT segmentation/measurement extension.
- AWS S3 dataset/artifact storage.
- Optional OpenAI-generated research explanation.
- Streamlit UI.
- Docker.

## Database
No relational database is required for the research prototype. S3 stores objects and JSON/CSV experiment artifacts.

## APIs
AWS S3 SDK/CLI; optional OpenAI API.

## Platform
Windows development; Docker/Linux deployment.

## Security
Private S3 bucket, least-privilege IAM, environment-based secrets, HTTPS/authentication for production.

## Medical limitation
The application is a research prototype. It does not provide clinical diagnosis. CT dimensions are only meaningful when the segmentation and physical spacing are valid.
