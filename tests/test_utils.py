import unittest
from market_knowledge_graph.utils.config_parser import load_config
from market_knowledge_graph.utils.logger import get_logger
from market_knowledge_graph.utils.exceptions import ApplicationError

class TestUtils(unittest.TestCase):
    def test_config(self):
        conf = load_config()
        self.assertIn('env', conf)
        
    def test_logger(self):
        logger = get_logger("test")
        self.assertIsNotNone(logger)
        
if __name__ == '__main__':
    unittest.main()
