# Complete Setup

## 1. Install Python
Use Python 3.11 on Windows.

## 2. Create environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. AWS CLI
Install AWS CLI v2, then:
```powershell
aws --version
aws configure
aws sts get-caller-identity
```
Enter your AWS access key, secret key, region and output format when prompted.

## 4. S3 bucket
Create a private bucket in AWS Console. Do not make medical datasets public.

## 5. Environment
```powershell
copy .env.example .env
```
Set S3_BUCKET to your bucket name.

## 6. X-ray data
Expected:
data/processed/xray/train/NORMAL
data/processed/xray/train/PNEUMONIA
data/processed/xray/validation/NORMAL
data/processed/xray/validation/PNEUMONIA
data/processed/xray/test/NORMAL
data/processed/xray/test/PNEUMONIA

## 7. Upload
```powershell
python scripts/s3_upload.py --local data/processed/xray --prefix datasets/xray
```

## 8. Train/evaluate
```powershell
python scripts/train_xray.py
python scripts/evaluate_xray.py
```

## 9. Upload model/results
```powershell
python scripts/s3_upload.py --local models --prefix models
python scripts/s3_upload.py --local results --prefix results
```

## 10. CT
Do not put a huge CT collection in S3 until its license, size and format are checked. LIDC-IDRI is large. Use a permitted subset first. Convert annotations to masks and preserve voxel spacing. The CT measurement code refuses to invent physical dimensions when spacing is unavailable.

## 11. OpenAI
Only set OPENAI_API_KEY after you have authorization to send the specific structured findings. Never put the key in source code.
