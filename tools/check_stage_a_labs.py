"""Verify teaching examples and streaming failure/count boundaries without a server."""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"labs"))
import stage_a_mechanisms as teaching
import streaming_client
import stage_a_plot


def stream(messages,done=True):
    data="".join("data: "+json.dumps(m)+"\n\n" for m in messages)
    if done:
        data+="data: [DONE]\n\n"
    return io.BytesIO(data.encode())


class TeachingChecks(unittest.TestCase):
    def test_scheduler_work_and_budget_with_partial_prefill(self):
        result=teaching.schedule([6,2,1],3,4)
        self.assertEqual(result["total_work_positions"],15)
        self.assertEqual([r["used"] for r in result["trace"]],[4,4,3,3,1])
        self.assertFalse(result["trace"][0]["allocations"][0]["sampled"])
        for prompts,outputs,budget in [([1],1,1),([1,9,3],5,2),([11,2],1,3)]:
            result=teaching.schedule(prompts,outputs,budget)
            self.assertEqual(result["total_work_positions"],sum(prompts)+len(prompts)*(outputs-1))
            self.assertTrue(all(r["used"]<=budget for r in result["trace"]))

    def test_tail_pool_and_ragged_positions_are_different_quantities(self):
        result=teaching.pages([5,8,9],4,1024,10)
        self.assertEqual([result[k] for k in ["logical_bytes","allocated_payload_bytes","pool_payload_bytes"]],[22*1024,28*1024,40*1024])
        self.assertFalse(teaching.pages([50],4,1024,10)["fits_pool"])
        result=teaching.inputs([1,3,2],[4,8,2])
        self.assertEqual(result["positions"],[4,8,9,10,2,3])
        self.assertEqual(result["query_start_loc"],[0,1,4,6])
        self.assertEqual(result["last_indices"],[0,3,5])

    def test_roofline_preserves_units(self):
        result=teaching.roofline(32,4096,4096,2,100,1000)
        self.assertEqual(result["minimum_traffic_bytes"],34078720)
        self.assertAlmostEqual(result["memory_lower_bound_us"],34.07872)


class StreamChecks(unittest.TestCase):
    def observe(self,messages,done=True):
        with patch("urllib.request.urlopen",return_value=stream(messages,done)):
            return streaming_client.observe("http://unit-test.invalid","fixture",[1,2],32)

    def test_empty_chunks_and_usage_do_not_become_token_timestamps(self):
        row=self.observe([
            {"choices":[{"index":0,"text":"","finish_reason":None}]},
            {"choices":[{"index":0,"text":"hello world","finish_reason":None}]},
            {"choices":[{"index":0,"text":"!","finish_reason":"length"}]},
            {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":32}},
        ])
        self.assertTrue(row["success"])
        self.assertTrue(row["counts_verified"])
        self.assertEqual(len(row["nonempty_chunk_events"]),2)
        self.assertNotIn("ITL",row)

    def test_truncated_stream_and_wrong_lengths_are_preserved_as_failures(self):
        good=[{"choices":[{"text":"hello","finish_reason":"length"}]}]
        row=self.observe(good,done=False)
        self.assertFalse(row["success"])
        self.assertIsNotNone(row["error"])
        wrong=good+[{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3}}]
        row=self.observe(wrong)
        self.assertFalse(row["success"])
        self.assertIn("output length",row["error"]["message"])

    def test_missing_usage_is_explicitly_unverified(self):
        row=self.observe([{"choices":[{"text":"hello","finish_reason":"length"}]}])
        self.assertTrue(row["success"])
        self.assertFalse(row["counts_verified"])
        self.assertIsNone(row["usage"])


class PlotChecks(unittest.TestCase):
    def fixture(self,tp,elapsed):
        return {
            "measurement":"completed offline batch wall time; test fixture only",
            "engine_arguments":{"tensor_parallel_size":tp,"model":"fixture",
                                "seed":17,"gpu_memory_utilization":0.5},
            "model_revision":"fixture-revision",
            "effective_settings":{"max_num_batched_tokens":2048},
            "rows":[{"input_tokens_per_request":128,"output_tokens_per_request":32,
                     "batch_requests":4,"batch_wall_seconds":elapsed,
                     "aggregate_output_tokens_per_second":128/elapsed,
                     "actual_input_counts":[128]*4,"actual_output_counts":[32]*4}],
        }

    def files(self,root,second):
        paths=[Path(root)/"tp1.json",Path(root)/"tp2.json"]
        for path,row in zip(paths,[self.fixture(1,2),second]):
            path.write_text(json.dumps(row))
        return paths

    def test_valid_completed_batches_produce_accessible_svg(self):
        with tempfile.TemporaryDirectory() as root:
            groups,cases=stage_a_plot.collect(self.files(root,self.fixture(2,1.5)))
            svg=ET.fromstring(stage_a_plot.plot(groups,cases))
            self.assertEqual(svg.attrib["role"],"img")
            self.assertEqual(len(groups),2)
            self.assertIn("not confidence intervals",ET.tostring(svg,encoding="unicode"))

    def test_workload_and_fixed_settings_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for field,value in [("actual_input_counts",[127]*4),
                                ("actual_output_counts",[31]*4),
                                ("aggregate_output_tokens_per_second",1000)]:
                second=self.fixture(2,1.5)
                second["rows"][0][field]=value
                with self.assertRaises(ValueError):
                    stage_a_plot.collect(self.files(root,second))
            for section,field,value in [("engine_arguments","seed",18),
                                        ("effective_settings","max_num_batched_tokens",4096)]:
                second=self.fixture(2,1.5)
                second[section][field]=value
                with self.assertRaises(ValueError):
                    stage_a_plot.collect(self.files(root,second))


if __name__=="__main__":
    unittest.main(verbosity=2)
