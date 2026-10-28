"""Verified external TurbL1 source; the pinned originals remain outside this repository."""
import hashlib
import importlib.util
from pathlib import Path

COMMIT = "58615378c13f3760bc0988e5e4d0f11e25a591b5"
HASHES = {
 "simvp": "80c98a3988ea22ecddcf553a0f1ec052a663e24ecbab4ceb9d81070bd95219d4",
 "convlstm": "7f7dd02f779a120d45f266f28553eebfaa154cef902845382eb8bb0ed5ec6952",
 "pastnet": "d07fcba8bd91a47c3e529fc78cab7882a6f6d97d4fafeb7d306b10c7216bd6bc"}

def load_upstream(settings):
 name = settings["architecture"]
 if settings.get("upstream_commit") != COMMIT:
  raise ValueError("Unpinned upstream architecture")
 path = Path(settings["upstream_root"])/"model_baselines"/(name+".py")
 if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
  raise ValueError("Upstream source checksum differs")
 spec = importlib.util.spec_from_file_location("neuralpom_external_"+name, path)
 module = importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module)
 return module
