#!/usr/bin/env python3
"""Download only three checksum-verified model files from the official pinned commit."""
import argparse, hashlib, sys, urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from neuralpom.ai_baselines.upstream import COMMIT, HASHES

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--destination",required=True)
 args=parser.parse_args();root=Path(args.destination)/"model_baselines";root.mkdir(parents=True,exist_ok=True)
 for name,expected in HASHES.items():
  target=root/(name+".py")
  if target.exists():
   if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise ValueError("Existing upstream file differs")
   continue
  url="https://raw.githubusercontent.com/Alexander-wu/TurbL1_AI4Science/"+COMMIT+"/model_baselines/"+name+".py"
  data=urllib.request.urlopen(url,timeout=60).read()
  if hashlib.sha256(data).hexdigest()!=expected:raise ValueError("Downloaded upstream file differs")
  with target.open("xb") as stream:stream.write(data)
 print("Verified upstream commit "+COMMIT)

if __name__=="__main__":main()
