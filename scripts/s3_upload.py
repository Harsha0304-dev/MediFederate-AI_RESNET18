import argparse,os
from pathlib import Path
from dotenv import load_dotenv
from src.medifederate.cloud import S3Store
load_dotenv()
p=argparse.ArgumentParser(); p.add_argument("--local",required=True); p.add_argument("--prefix",required=True); a=p.parse_args()
s=S3Store()
for f in Path(a.local).rglob("*"):
    if f.is_file(): s.upload_file(f,f"{a.prefix.rstrip('/')}/{f.relative_to(a.local).as_posix()}"); print("uploaded",f)
