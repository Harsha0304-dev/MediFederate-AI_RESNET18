import argparse,os
from dotenv import load_dotenv
from src.medifederate.cloud import S3Store
load_dotenv()
p=argparse.ArgumentParser(); p.add_argument("--prefix",required=True); p.add_argument("--output",required=True); a=p.parse_args()
S3Store().download_prefix(a.prefix,a.output); print("download complete")
