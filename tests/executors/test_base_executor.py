import unittest
from unittest.mock import patch, MagicMock
from agent_core.executors.base_executor import BaseExecutor
from agent_core.models.model_registry import ModelRegistry


class TestBaseExecutor(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.process.return_value = "test response"
        self.model_patcher = patch.object(
            ModelRegistry, 
            'get_model',
            return_value=self.mock_model
        )
        self.mock_get_model = self.model_patcher.start()

    def tearDown(self):
        self.model_patcher.stop()

    def test_init_defaults(self):
        """Test initialization with default parameters"""
        executor = BaseExecutor()
        self.assertEqual(executor.model_name, 'gemini-1.5-flash-002')
        self.assertEqual(executor.name, "BaseExecutor")

    def test_init_with_parameters(self):
        """Test initialization with custom parameters"""
        executor = BaseExecutor(model_name="test_model")
        self.assertEqual(executor.model_name, "test_model")

    def test_execute_default_model(self):
        """Test execute with default model"""
        executor = BaseExecutor(model_name="default_model")
        result = executor.execute("test prompt")
        
        self.assertEqual(self.mock_get_model.call_count, 2)
        self.mock_model.process.assert_called_once_with("test prompt")
        self.assertEqual(result, "test response")

    def test_execute_override_model(self):
        """Test execute with overridden model name"""
        executor = BaseExecutor(model_name="default_model")
        result = executor.execute("test prompt", model_name="custom_model")
        
        self.assertEqual(self.mock_get_model.call_count, 2)
        self.mock_model.process.assert_called_once_with("test prompt")
        self.assertEqual(result, "test response")

    def test_execute_empty_prompt(self):
        """Test execute with empty prompt"""
        executor = BaseExecutor()
        result = executor.execute("")
        
        self.mock_model.process.assert_called_once_with("")
        self.assertEqual(result, "test response")

    def test_execute_logging(self):
        """Test debug logging during execution"""
        with self.assertLogs('BaseExecutor', level='DEBUG') as cm:
            executor = BaseExecutor()
            executor.execute("test prompt")
            
            self.assertIn('BaseExecutor uses model: gemini-1.5-flash-002', cm.output[0])


if __name__ == '__main__':
    unittest.main()
