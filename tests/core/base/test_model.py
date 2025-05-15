import pytest
import torch
from horguesc.core.base.model import BaseModel

class TestBaseModel:
    """Tests for the BaseModel class."""
    
    def test_abstract_methods_must_be_implemented(self):
        """Test that all abstract methods must be implemented by child classes."""
        # Abstract class can't be instantiated directly
        with pytest.raises(TypeError):
            BaseModel({})
    
    def test_child_class_implementation(self):
        """Test that a child class properly implementing all abstract methods works."""
        
        class ConcreteModel(BaseModel):
            """A concrete implementation of BaseModel for testing."""
            
            def forward(self, inputs):
                return {"output": inputs["input"] * 2}
            
            def compute_loss(self, outputs, targets):
                return torch.mean(torch.abs(outputs["output"] - targets["target"]))
            
            def compute_metrics(self, outputs, targets):
                # Add a simple accuracy metric
                correct = torch.isclose(outputs["output"], targets["target"])
                accuracy = torch.mean(correct.float())
                return {"accuracy": accuracy.item()}
            
            def get_name(self):
                return "TestModel"
        
        # Should instantiate without errors
        config = {"test": "config"}
        model = ConcreteModel(config)
        
        # Test forward method
        inputs = {"input": torch.tensor([1.0, 2.0, 3.0])}
        outputs = model(inputs)
        assert "output" in outputs
        assert torch.allclose(outputs["output"], torch.tensor([2.0, 4.0, 6.0]))
        
        # Test compute_loss method
        targets = {"target": torch.tensor([2.0, 3.0, 4.0])}
        loss = model.compute_loss(outputs, targets)
        expected_loss = torch.mean(torch.tensor([0.0, 1.0, 2.0]))  # |2-2|, |4-3|, |6-4|
        assert torch.isclose(loss, expected_loss)
        
        # Test compute_metrics method
        metrics = model.compute_metrics(outputs, targets)
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)
        
        # Test get_name method
        assert model.get_name() == "TestModel"
    
    def test_encoder_parameter(self):
        """Test that encoder parameter is correctly stored."""
        
        class ConcreteModel(BaseModel):
            def forward(self, inputs): return {}
            def compute_loss(self, outputs, targets): return torch.tensor(0.0)
            def compute_metrics(self, outputs, targets): return {"dummy_metric": 0.0}
            def get_name(self): return "TestModel"
        
        # Create a simple mock encoder
        mock_encoder = torch.nn.Linear(10, 5)
        
        # Instantiate model with encoder
        model = ConcreteModel({}, encoder=mock_encoder)
        
        # Verify encoder is correctly stored
        assert model.encoder is mock_encoder