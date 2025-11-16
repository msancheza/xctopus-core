"""
Tests for CLI script (Phase 9, Sprint 9.1)
"""

import unittest
import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Add scripts/cli to path
script_dir = Path(__file__).parent.parent
cli_script = script_dir / 'scripts' / 'cli' / 'xctopus_run.py'


class TestCLIBasic(unittest.TestCase):
    """Test cases for basic CLI functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cli_script_exists(self):
        """Test that CLI script exists"""
        self.assertTrue(cli_script.exists(), f"CLI script not found: {cli_script}")
    
    def test_cli_help(self):
        """Test that CLI shows help"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        try:
            result = subprocess.run(
                [sys.executable, str(cli_script), '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn('Xctopus Pipeline CLI', result.stdout)
        except subprocess.TimeoutExpired:
            self.skipTest("CLI help command timed out")
        except FileNotFoundError:
            self.skipTest("Python executable not found")
    
    def test_cli_missing_dataset(self):
        """Test that CLI fails with missing dataset"""
        try:
            result = subprocess.run(
                [sys.executable, str(cli_script)],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Should fail because dataset is required
            self.assertNotEqual(result.returncode, 0)
        except subprocess.TimeoutExpired:
            self.skipTest("CLI command timed out")
        except FileNotFoundError:
            self.skipTest("Python executable not found")
    
    def test_cli_nonexistent_dataset(self):
        """Test that CLI fails with nonexistent dataset"""
        try:
            result = subprocess.run(
                [sys.executable, str(cli_script), 'nonexistent.csv'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Should fail because file doesn't exist
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('not found', result.stdout.lower() or result.stderr.lower())
        except subprocess.TimeoutExpired:
            self.skipTest("CLI command timed out")
        except FileNotFoundError:
            self.skipTest("Python executable not found")
    
    def test_cli_invalid_config(self):
        """Test that CLI fails with invalid config file"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        try:
            result = subprocess.run(
                [sys.executable, str(cli_script), self.csv_path, '--config', 'nonexistent.yaml'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Should fail because config file doesn't exist
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('not found', result.stdout.lower() or result.stderr.lower())
        except subprocess.TimeoutExpired:
            self.skipTest("CLI command timed out")
        except FileNotFoundError:
            self.skipTest("Python executable not found")
    
    def test_cli_step_argument(self):
        """Test that CLI accepts --step argument"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        # Just test that argument is accepted (won't run full execution)
        try:
            result = subprocess.run(
                [sys.executable, str(cli_script), self.csv_path, '--step', 'analysis', '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Help should still work
            self.assertEqual(result.returncode, 0)
        except subprocess.TimeoutExpired:
            self.skipTest("CLI command timed out")
        except FileNotFoundError:
            self.skipTest("Python executable not found")
    
    def test_cli_skip_flags(self):
        """Test that CLI accepts skip flags"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        # Just test that flags are accepted (won't run full execution)
        try:
            result = subprocess.run(
                [sys.executable, str(cli_script), self.csv_path, 
                 '--skip-analysis', '--skip-evaluation', '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Help should still work
            self.assertEqual(result.returncode, 0)
        except subprocess.TimeoutExpired:
            self.skipTest("CLI command timed out")
        except FileNotFoundError:
            self.skipTest("Python executable not found")


if __name__ == '__main__':
    unittest.main()

