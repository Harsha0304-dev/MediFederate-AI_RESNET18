import os
from pathlib import Path
import boto3

class S3Store:
    def __init__(self,bucket=None,region=None):
        self.bucket=bucket or os.getenv('S3_BUCKET')
        self.region=region or os.getenv('AWS_REGION','eu-north-1')
        if not self.bucket: raise ValueError('Set S3_BUCKET in .env')
        self.s3=boto3.client('s3',region_name=self.region)
    def upload_file(self,local,key): self.s3.upload_file(str(local),self.bucket,key)
    def sync_directory(self,local_dir,prefix):
        local=Path(local_dir)
        for p in local.rglob('*'):
            if p.is_file(): self.upload_file(p, f"{prefix.rstrip('/')}/{p.relative_to(local).as_posix()}")
    def download_prefix(self,prefix,out):
        out=Path(out); out.mkdir(parents=True,exist_ok=True)
        paginator=self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket,Prefix=prefix):
            for obj in page.get('Contents',[]):
                key=obj['Key']; rel=key[len(prefix):].lstrip('/')
                if rel: dest=out/rel; dest.parent.mkdir(parents=True,exist_ok=True); self.s3.download_file(self.bucket,key,str(dest))
