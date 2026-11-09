import unittest,json,pathlib
class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules=json.loads(pathlib.Path('network/nacl.json').read_text(encoding='utf-8-sig'))
    def test_directions(self):
        self.assertIn('inbound',self.rules);self.assertIn('outbound',self.rules)
    def test_deny_rule(self):
        self.assertTrue(any(r['action']=='deny' and r['cidr']=='198.51.100.0/24' for r in self.rules['inbound']))
    def test_port_ranges(self):
        self.assertTrue(all(0<=r['port_from']<=r['port_to']<=65535 for d in ('inbound','outbound') for r in self.rules[d]))
