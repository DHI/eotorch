"""
Integration tests for DINOv3 + UPerNet semantic segmentation workflow.

Tests focus on:
1. Model registry is set up correctly
2. SEG_MODEL_MAPPING contains the DINOv3UPerNet model
3. Configuration structure is correct
"""

import pytest
from eotorch.models import SEG_MODEL_MAPPING, DINOv3UPerNet


class TestDINOv3UPerNetRegistry:
    """Test that DINOv3UPerNet is properly registered."""

    def test_seg_model_mapping_exists(self):
        """Test that SEG_MODEL_MAPPING exists."""
        assert SEG_MODEL_MAPPING is not None
        assert isinstance(SEG_MODEL_MAPPING, dict)

    def test_dinov3_upernet_in_mapping(self):
        """Test that dinov3_upernet is registered in SEG_MODEL_MAPPING."""
        assert "dinov3_upernet" in SEG_MODEL_MAPPING
        assert SEG_MODEL_MAPPING["dinov3_upernet"] == DINOv3UPerNet

    def test_dinov3_upernet_is_module(self):
        """Test that DINOv3UPerNet is a proper module class."""
        import torch.nn as nn
        assert issubclass(DINOv3UPerNet, nn.Module)


class TestDINOv3UPerNetModelProperties:
    """Test DINOv3UPerNet model class properties."""

    def test_dinov3_upernet_has_required_methods(self):
        """Test that DINOv3UPerNet has required PyTorch methods."""
        required_methods = ["__init__", "forward"]
        for method in required_methods:
            assert hasattr(DINOv3UPerNet, method), f"Missing {method}"

    def test_dinov3_upernet_init_signature(self):
        """Test that __init__ has expected parameters."""
        import inspect
        sig = inspect.signature(DINOv3UPerNet.__init__)
        params = list(sig.parameters.keys())
        
        required_params = ["self", "num_classes", "dinov3_model_name"]
        for param in required_params:
            assert param in params, f"Missing parameter: {param}"


class TestWorkflowDocumentation:
    """Test that workflow documentation exists."""

    def test_workflow_readme_exists(self):
        """Test that DINOv3+UPerNet workflow documentation exists."""
        from pathlib import Path
        workflow_file = Path("d:/NISI/github/eotorch/DINOV3_UPERNET_WORKFLOW.md")
        assert workflow_file.exists(), "Workflow documentation not found"

    def test_workflow_notebook_exists(self):
        """Test that example notebook exists."""
        from pathlib import Path
        notebook_file = Path("d:/NISI/github/eotorch/notebooks/dinov3_upernet_workflow.ipynb")
        assert notebook_file.exists(), "Example notebook not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
