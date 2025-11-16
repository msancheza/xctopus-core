"""
Tests for PipelineConfig Phase 2 features:
- YAML loading
- Auto-detection of text columns
- Column validation
"""

import unittest
import os
import tempfile
import pandas as pd
from xctopus.pipeline.config import PipelineConfig


class TestPipelineConfigYAML(unittest.TestCase):
    """Test cases for YAML loading"""
    
    def setUp(self):
        """Create temporary test files"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.yaml_path = os.path.join(self.test_dir, 'test_config_yaml.yaml')
    
    def test_from_yaml_file_exists(self):
        """Test loading config from existing YAML file"""
        if not os.path.exists(self.yaml_path):
            self.skipTest(f"Test YAML file not found: {self.yaml_path}")
        
        config = PipelineConfig.from_yaml(self.yaml_path)
        
        # Check dataset config
        self.assertEqual(config.TEXT_COLUMNS, ["title", "abstract"])
        self.assertEqual(config.JOIN_WITH, "\n")
        self.assertIsNone(config.LABEL_COLUMN)
        self.assertEqual(config.ID_COLUMN, "uuid")
        self.assertTrue(config.DROP_EMPTY)
        
        # Check embeddings config
        self.assertEqual(config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
        self.assertEqual(config.MAX_LENGTH, 256)
        self.assertTrue(config.NORMALIZE_EMBEDDINGS)
        
        # Check pipeline config
        self.assertEqual(config.NUM_EPOCHS, 10)
        self.assertEqual(config.LEARNING_RATE, 0.0001)
        self.assertFalse(config.ENABLE_FINE_TUNE_LARGE)
        self.assertTrue(config.ENABLE_AUTO_UPDATE)
        
        # Check clustering config
        self.assertEqual(config.MIN_CLUSTER_SIZE, 10)
        self.assertEqual(config.ORPHAN_THRESHOLD, 5)
        
        # Check evaluation config
        self.assertFalse(config.ENABLE_AUDIT)
        self.assertTrue(config.ENABLE_PERFORMANCE_EVAL)
    
    def test_from_yaml_file_not_found(self):
        """Test that loading non-existent file raises error"""
        with self.assertRaises(FileNotFoundError):
            PipelineConfig.from_yaml('nonexistent_config.yaml')
    
    def test_from_yaml_partial_config(self):
        """Test loading YAML with only some sections"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
dataset:
  text_columns: ["title"]
pipeline:
  epochs: 20
""")
            temp_path = f.name
        
        try:
            config = PipelineConfig.from_yaml(temp_path)
            self.assertEqual(config.TEXT_COLUMNS, ["title"])
            self.assertEqual(config.NUM_EPOCHS, 20)
            # Other values should be defaults
            self.assertEqual(config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
        finally:
            os.unlink(temp_path)


class TestColumnDetection(unittest.TestCase):
    """Test cases for text column detection"""
    
    def setUp(self):
        """Create temporary test CSV files"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_detect_text_columns_by_name(self):
        """Test detection of columns by common names"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        detected = PipelineConfig.detect_text_columns(self.csv_path)
        
        # Should detect 'title' and 'abstract' (common names)
        self.assertGreater(len(detected), 0)
        self.assertIn('title', detected)
        self.assertIn('abstract', detected)
    
    def test_detect_text_columns_by_type(self):
        """Test detection by data type when name doesn't match"""
        # Create CSV with non-standard column names
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n")
            f.write('"Some text content","More text here","123"\n')
            f.write('"Another text","Yet more text","456"\n')
            temp_path = f.name
        
        try:
            detected = PipelineConfig.detect_text_columns(temp_path)
            # Should detect object columns with text
            self.assertGreater(len(detected), 0)
        finally:
            os.unlink(temp_path)
    
    def test_detect_text_columns_no_text(self):
        """Test detection when no text columns exist"""
        # Create CSV with only numeric columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n")
            f.write("1,2\n")
            f.write("3,4\n")
            temp_path = f.name
        
        try:
            detected = PipelineConfig.detect_text_columns(temp_path)
            self.assertEqual(detected, [])
        finally:
            os.unlink(temp_path)
    
    def test_detect_text_columns_invalid_file(self):
        """Test that invalid file raises error"""
        with self.assertRaises(ValueError):
            PipelineConfig.detect_text_columns('nonexistent.csv')
    
    def test_suggest_text_columns_auto_detect(self):
        """Test suggest_text_columns with auto-detection"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        suggested = PipelineConfig.suggest_text_columns(self.csv_path)
        self.assertGreater(len(suggested), 0)
        self.assertIn('title', suggested)
    
    def test_suggest_text_columns_manual_fallback(self):
        """Test suggest_text_columns with manual suggestions"""
        # Create CSV without common text column names
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("custom_col1,custom_col2\n")
            f.write('"Text here","More text"\n')
            temp_path = f.name
        
        try:
            # With manual suggestions
            suggested = PipelineConfig.suggest_text_columns(
                temp_path, 
                suggested=['custom_col1']
            )
            self.assertEqual(suggested, ['custom_col1'])
        finally:
            os.unlink(temp_path)
    
    def test_suggest_text_columns_final_fallback(self):
        """Test suggest_text_columns final fallback to 'text' column"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,other\n")
            f.write('"Some text","Other data"\n')
            temp_path = f.name
        
        try:
            suggested = PipelineConfig.suggest_text_columns(temp_path)
            self.assertEqual(suggested, ['text'])
        finally:
            os.unlink(temp_path)
    
    def test_get_available_columns(self):
        """Test _get_available_columns helper"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        columns = PipelineConfig._get_available_columns(self.csv_path)
        self.assertIsInstance(columns, list)
        self.assertGreater(len(columns), 0)
        self.assertIn('title', columns)
        self.assertIn('abstract', columns)


class TestColumnValidation(unittest.TestCase):
    """Test cases for column validation"""
    
    def setUp(self):
        """Create temporary test CSV files"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_validate_columns_all_exist(self):
        """Test validation when all columns exist"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        result = PipelineConfig.validate_dataset_columns(
            self.csv_path,
            text_columns=['title', 'abstract'],
            label_column='label',
            id_column='id'
        )
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertEqual(len(result['warnings']), 0)
        self.assertIn('title', result['available_columns'])
    
    def test_validate_columns_missing_warning(self):
        """Test validation with missing columns (warning mode)"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        result = PipelineConfig.validate_dataset_columns(
            self.csv_path,
            text_columns=['title', 'nonexistent'],
            strict=False
        )
        
        self.assertTrue(result['valid'])  # Still valid (warnings only)
        self.assertEqual(len(result['errors']), 0)
        self.assertGreater(len(result['warnings']), 0)
        self.assertIn('nonexistent', result['warnings'][0])
    
    def test_validate_columns_missing_strict(self):
        """Test validation with missing columns (strict mode)"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        with self.assertRaises(ValueError) as context:
            PipelineConfig.validate_dataset_columns(
                self.csv_path,
                text_columns=['title', 'nonexistent'],
                strict=True
            )
        
        self.assertIn('not found', str(context.exception))
    
    def test_validate_columns_invalid_file(self):
        """Test validation with invalid file"""
        result = PipelineConfig.validate_dataset_columns(
            'nonexistent.csv',
            text_columns=['title'],
            strict=False
        )
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_validate_columns_no_specifications(self):
        """Test validation with no column specifications"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        result = PipelineConfig.validate_dataset_columns(self.csv_path)
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertEqual(len(result['warnings']), 0)


if __name__ == '__main__':
    unittest.main()

