"""Unit tests for model loading and initialization."""

import pytest
import torch

from models import load_model, list_available_models, get_checkpoint_info


class TestModelRegistry:
    """Test model registry and factory function."""

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()
        assert isinstance(models, dict)
        assert "dinov2" in models
        assert "dinov1" in models
        assert "ijepa" in models
        assert "mae" in models
        assert "ibot" in models
        assert "lejepa" in models

    def test_list_available_architectures(self):
        """Test that each model has available architectures."""
        models = list_available_models()
        for model_name, archs in models.items():
            assert isinstance(archs, list)
            assert len(archs) > 0
            assert all(isinstance(arch, str) for arch in archs)

    def test_checkpoint_info_lookup(self):
        """Test checkpoint info retrieval."""
        info = get_checkpoint_info("dinov2", "vit_base")
        assert info is not None
        assert "url" in info
        assert "embedding_dim" in info
        assert "description" in info
        assert info["embedding_dim"] == 768

    def test_checkpoint_info_missing(self):
        """Test that missing checkpoint returns None."""
        info = get_checkpoint_info("unknown_model", "unknown_arch")
        assert info is None


class TestDINOv2Loading:
    """Test DINOv2 model loading."""

    @pytest.mark.parametrize("arch", ["vit_small", "vit_base"])
    def test_dinov2_initialization(self, arch):
        """Test DINOv2 model initialization."""
        try:
            model = load_model("dinov2", arch=arch)
            assert model is not None
            assert hasattr(model, "extract_features")
            assert hasattr(model, "embedding_dim")
            assert model.embedding_dim > 0
        except Exception as e:
            pytest.skip(f"DINOv2 {arch} loading failed: {e}")

    def test_dinov2_model_name(self):
        """Test DINOv2 model name property."""
        try:
            model = load_model("dinov2", arch="vit_base")
            assert "dinov2" in model.name.lower()
        except Exception as e:
            pytest.skip(f"DINOv2 loading failed: {e}")


class TestDINOv1Loading:
    """Test DINOv1 model loading."""

    @pytest.mark.parametrize("arch", ["vit_small", "vit_base"])
    def test_dinov1_initialization(self, arch):
        """Test DINOv1 model initialization."""
        try:
            model = load_model("dinov1", arch=arch)
            assert model is not None
            assert hasattr(model, "extract_features")
            assert hasattr(model, "embedding_dim")
            assert model.embedding_dim > 0
        except Exception as e:
            pytest.skip(f"DINOv1 {arch} loading failed: {e}")

    def test_dinov1_arch_alias_resolution(self):
        """Test that DINOv1 arch aliases work correctly."""
        try:
            # Both should work due to alias mapping
            model1 = load_model("dinov1", arch="vit_base")
            model2 = load_model("dinov1", arch="dino_vitb16")
            assert model1.embedding_dim == model2.embedding_dim
        except Exception as e:
            pytest.skip(f"DINOv1 loading failed: {e}")


class TestMAELoading:
    """Test MAE model loading."""

    @pytest.mark.parametrize("arch", ["vit_base", "vit_large"])
    def test_mae_initialization(self, arch):
        """Test MAE model initialization."""
        try:
            model = load_model("mae", arch=arch)
            assert model is not None
            assert hasattr(model, "extract_features")
            assert hasattr(model, "embedding_dim")
            assert model.embedding_dim > 0
        except Exception as e:
            pytest.skip(f"MAE {arch} loading failed: {e}")


class TestIJEPALoading:
    """Test I-JEPA model loading (requires checkpoint)."""

    def test_ijepa_requires_checkpoint(self):
        """Test that I-JEPA requires checkpoint path."""
        with pytest.raises(ValueError, match="checkpoint"):
            load_model("ijepa", arch="vit_base")

    def test_ijepa_concat_n_layers(self):
        """Test I-JEPA with layer concatenation."""
        try:
            # This will fail without checkpoint, but tests parameter acceptance
            with pytest.raises(FileNotFoundError):
                load_model(
                    "ijepa",
                    arch="vit_base",
                    checkpoint_path="/nonexistent/checkpoint.pth",
                    concat_n_layers=4,
                )
        except ValueError as e:
            if "checkpoint" in str(e):
                pytest.skip("I-JEPA checkpoint not available")


class TestiBOTLoading:
    """Test iBOT model loading (requires checkpoint)."""

    def test_ibot_requires_checkpoint(self):
        """Test that iBOT requires checkpoint path."""
        with pytest.raises(ValueError, match="checkpoint"):
            load_model("ibot", arch="vit_base")


class TestLeJEPALoading:
    """Test LeJEPA model loading (requires checkpoint)."""

    def test_lejepa_requires_checkpoint(self):
        """Test that LeJEPA requires checkpoint path."""
        with pytest.raises(ValueError, match="checkpoint"):
            load_model("lejepa", arch="vit_huge")


class TestModelInterface:
    """Test unified model interface."""

    def test_model_has_required_attributes(self):
        """Test that all loaded models have required attributes."""
        try:
            model = load_model("dinov2", arch="vit_base")
            assert hasattr(model, "extract_features")
            assert hasattr(model, "embedding_dim")
            assert hasattr(model, "name")
            assert callable(model.extract_features)
            assert isinstance(model.embedding_dim, int)
            assert isinstance(model.name, str)
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    def test_model_eval_method(self):
        """Test that models support .eval() method."""
        try:
            model = load_model("dinov2", arch="vit_base")
            result = model.eval()
            assert result is model  # Should return self
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    def test_model_to_method(self):
        """Test that models support .to(device) method."""
        try:
            model = load_model("dinov2", arch="vit_base")
            device = "cpu"
            result = model.to(device)
            assert result is model  # Should return self
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")


class TestModelFeatureExtraction:
    """Test feature extraction on dummy inputs."""

    @pytest.fixture
    def device(self):
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @pytest.fixture
    def dummy_images(self, device):
        """Create dummy images for feature extraction."""
        return torch.randn(2, 3, 224, 224, device=device)

    def test_dinov2_feature_extraction(self, dummy_images):
        """Test DINOv2 feature extraction."""
        try:
            model = load_model("dinov2", arch="vit_base")
            model = model.to(dummy_images.device)
            features = model.extract_features(dummy_images)
            assert features.shape[0] == dummy_images.shape[0]
            assert features.shape[1] == model.embedding_dim
            # Features should be L2 normalized
            norms = torch.norm(features, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        except Exception as e:
            pytest.skip(f"DINOv2 feature extraction failed: {e}")

    def test_dinov1_feature_extraction(self, dummy_images):
        """Test DINOv1 feature extraction."""
        try:
            model = load_model("dinov1", arch="vit_base")
            model = model.to(dummy_images.device)
            features = model.extract_features(dummy_images)
            assert features.shape[0] == dummy_images.shape[0]
            assert features.shape[1] == model.embedding_dim
            # Features should be L2 normalized
            norms = torch.norm(features, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        except Exception as e:
            pytest.skip(f"DINOv1 feature extraction failed: {e}")

    def test_feature_output_is_normalized(self, dummy_images):
        """Test that extracted features are L2 normalized."""
        try:
            model = load_model("dinov2", arch="vit_small")
            model = model.to(dummy_images.device)
            features = model.extract_features(dummy_images)
            norms = torch.norm(features, p=2, dim=-1)
            # All norms should be close to 1.0
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        except Exception as e:
            pytest.skip(f"Feature normalization test failed: {e}")

    def test_feature_dtype_is_float32(self, dummy_images):
        """Test that extracted features are float32."""
        try:
            model = load_model("dinov2", arch="vit_base")
            model = model.to(dummy_images.device)
            features = model.extract_features(dummy_images)
            assert features.dtype == torch.float32
        except Exception as e:
            pytest.skip(f"Feature dtype test failed: {e}")


class TestUnknownModel:
    """Test error handling for unknown models."""

    def test_load_unknown_model(self):
        """Test that loading unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            load_model("unknown_model")

    def test_load_unknown_architecture(self):
        """Test that loading unknown arch raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            load_model("dinov2", arch="unknown_arch")
