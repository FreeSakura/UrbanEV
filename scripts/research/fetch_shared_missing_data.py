"""Fetch the two public UCI archives for the shared-missing-events pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

SOURCES={
    'beijing_original.zip':'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip',
    'household_original.zip':'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip',
}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);records=[]
    for name,url in SOURCES.items():
        path=a.output/name
        if not path.exists() or not zipfile.is_zipfile(path):
            with urllib.request.urlopen(url,timeout=120) as response,path.open('wb') as destination:
                shutil.copyfileobj(response,destination)
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:raise ValueError(f'Archive CRC mismatch: {name}')
        records.append({'file':name,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
        print(f'{name}: {path.stat().st_size} bytes',flush=True)
    (a.output/'download_manifest.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
