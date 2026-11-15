import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.scorers import contains,exact_numeric,json_schema_valid,json_field_f1,tool_call_match,refusal_v1


class ScorerChecks(unittest.TestCase):
    def test_contains_boundaries(self):
        self.assertEqual(contains('nice','ice'),0);self.assertEqual(contains('14','4'),0)
        self.assertEqual(contains('ICE!', ['ice','frozen water']),1)
    def test_number(self):
        self.assertEqual(exact_numeric('#### 1,000','1000'),1)
        self.assertEqual(exact_numeric('4 or 5',4),0)
        self.assertEqual(exact_numeric('NaN','NaN'),0)
    def test_json(self):
        self.assertEqual(json_schema_valid('{"x":1}',{'type':'object','required':['x']}),1)
        self.assertEqual(json_schema_valid('{}',{'type':'object','required':['x']}),0)
        self.assertEqual(json_field_f1('{"x":1,"z":2}',{'x':1}),2/3)
    def test_tool(self):
        gold={'name':'add','arguments':{'x':1}}
        self.assertEqual(tool_call_match('{"arguments":{"x":1},"name":"add"}',gold),1)
        self.assertEqual(tool_call_match('{"arguments":{"x":2},"name":"add"}',gold),0)
    def test_refusal(self):
        self.assertEqual(refusal_v1("I cannot help with that request."),1)
        self.assertEqual(refusal_v1('Here are safe washing instructions.'),0)

    def test_invalid_json_constants_and_duplicate_keys(self):
        for answer in ['{"x":NaN}','{"x":Infinity}','{"x":1,"x":2}']:
            self.assertEqual(json_schema_valid(answer,{'type':'object'}),0)
            self.assertEqual(json_field_f1(answer,{'x':2}),0)
        self.assertEqual(tool_call_match('{"name":"wrong","name":"add","arguments":{}}',{'name':'add','arguments':{}}),0)
