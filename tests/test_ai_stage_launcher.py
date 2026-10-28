"""Check that a failed prerequisite never starts later scientific stages."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('ai_stage_launcher',Path(__file__).resolve().parents[1]/'scripts/run_ai_stages.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


class StageLauncherTests(unittest.TestCase):
    def execute(self, failing=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run'
            config=Path(directory)/'config.json'
            config.write_text(json.dumps(dict(data_root='/unused-reference',run_root=str(root))))
            launched=[]
            class Process:
                pid=12345
                def __init__(self,command,**kwargs):
                    self.stage='prepare' if 'prepare' in command else command[command.index('--stage')+1]
                    launched.append(self.stage)
                def wait(self):
                    if self.stage==failing:return 3
                    if self.stage=='prepare':
                        (root/'assets').mkdir();(root/'assets/state_stats.npz').write_bytes(b'fixture')
                    else:
                        (root/self.stage).mkdir();(root/self.stage/'COMPLETE.json').write_text('{}')
                        (root/'frozen').mkdir(exist_ok=True);(root/f'frozen/{self.stage}.pt').write_bytes(b'fixture')
                    return 0
            with patch.object(sys,'argv',['launcher','--config',str(config),'--tracking','offline']),patch.object(launcher.subprocess,'Popen',Process):
                if failing:
                    with self.assertRaises(RuntimeError):launcher.main()
                else:launcher.main()
            return launched,json.loads((root/'pipeline.json').read_text())

    def test_success_chains_only_two_training_stages(self):
        stages,state=self.execute()
        self.assertEqual(stages,['prepare','pretrain','posttrain'])
        self.assertEqual(state['status'],'complete')

    def test_pretrain_failure_prevents_posttrain(self):
        stages,state=self.execute('pretrain')
        self.assertEqual(stages,['prepare','pretrain'])
        self.assertEqual(state['status'],'failed')
        self.assertEqual(state['completed_stages'],['prepare'])

    def test_preparation_failure_prevents_all_training(self):
        stages,state=self.execute('prepare')
        self.assertEqual(stages,['prepare'])
        self.assertEqual(state['status'],'failed')
