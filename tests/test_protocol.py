import json, tempfile, unittest
from pathlib import Path
from benchmark.experiment import planned_api_calls, METHODS, seconds_until_next_call
from benchmark.scene import generate_task, SCENARIOS
from benchmark.provider import parse_decision, DECISION_SCHEMA, Provider
from benchmark.checkpoints import save_checkpoint, load_checkpoint, checkpoint_complete, checkpoint_has_api_result

class TestProtocol(unittest.TestCase):
    def test_exact_budget(self): self.assertEqual(planned_api_calls(50),50)
    def test_four_methods(self): self.assertEqual(len(METHODS),4)
    def test_new_scene_ids_do_not_overlap_v11(self):
        old={generate_task(i,20261121)['seed'] for i in range(1,21)}
        new={generate_task(i,20261121)['seed'] for i in range(21,71)}
        self.assertEqual(len(new),50); self.assertFalse(old & new)
    def test_scenario_distribution(self):
        counts={s:0 for s in SCENARIOS}
        for i in range(21,71): counts[generate_task(i,20261121)['scenario']]+=1
        self.assertEqual(sorted(counts.values()),[12,12,13,13])
    def test_structured_schema(self):
        self.assertEqual(set(DECISION_SCHEMA['required']),{'object_color','goal_color','route_hint'})
    def test_output_budget_config(self):
        p=Provider('local','', 'm', 1, 10, 90, 4096); self.assertEqual(p.max_output_tokens,4096)
    def test_json_parser(self):
        t=generate_task(21,20261121); d=parse_decision('{"object_color":"red","goal_color":"green","route_hint":"upper"}',t); self.assertEqual(d['route_hint'],'upper')
    def test_persisted_rate_wait(self):
        self.assertAlmostEqual(seconds_until_next_call(1000,1,1.5,now=1000),61.5)
        self.assertEqual(seconds_until_next_call(1000,1,1.5,now=1062),0.0)
    def test_checkpoint_api_reuse_state(self):
        with tempfile.TemporaryDirectory() as td:
            cp={'status':'api_complete','task':{'scene_id':21},'api_done':True,'decision':{'object_color':'red'},'call':{'latency_s':1.2},'method_rows':[]}
            save_checkpoint(td,21,cp); r=load_checkpoint(td,21)
            self.assertTrue(checkpoint_has_api_result(r)); self.assertFalse(checkpoint_complete(r)); self.assertEqual(r['decision']['object_color'],'red')
    def test_checkpoint_complete(self):
        with tempfile.TemporaryDirectory() as td:
            cp={'status':'complete','task':{'scene_id':21},'api_done':True,'decision':{},'call':{},'method_rows':[{}, {}, {}, {}]}
            save_checkpoint(td,21,cp); self.assertTrue(checkpoint_complete(load_checkpoint(td,21)))

if __name__=='__main__': unittest.main()
