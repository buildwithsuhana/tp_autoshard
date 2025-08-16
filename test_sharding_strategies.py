#!/usr/bin/env python3
"""
Test different sharding strategies with Keras Tensor Parallel
"""

import numpy as np
import keras
from keras import layers, Model
from src.tensor_parallel_keras import TensorParallelKeras

def create_test_model():
    """Create a test model with multiple Dense layers."""
    inputs = keras.Input(shape=(100,), name="input")
    
    # First Dense layer
    x = layers.Dense(512, activation='relu', name="dense1")(inputs)
    
    # Second Dense layer
    x = layers.Dense(256, activation='relu', name="dense2")(x)
    
    # Third Dense layer
    x = layers.Dense(128, activation='relu', name="dense3")(x)
    
    # Output layer
    outputs = layers.Dense(10, activation='softmax', name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="test_model")
    return model

def test_sharding_strategy(strategy: str):
    """Test a specific sharding strategy."""
    print(f"\n🧪 Testing {strategy.upper()} sharding strategy...")
    print("=" * 50)
    
    # Create model
    model = create_test_model()
    total_params = sum(w.shape.num_elements() for w in model.weights)
    print(f"📊 Original model parameters: {total_params:,}")
    
    try:
        # Create tensor parallel model with specific strategy
        tp_model = TensorParallelKeras(
            model,
            device_ids=["cpu", "cpu"],
            sharded=True,
            sharding_strategy=strategy
        )
        print(f"✅ {strategy.upper()} strategy created successfully!")
        
        # Check sharding
        print(f"📊 Number of shards: {len(tp_model.model_shards)}")
        for i, shard in enumerate(tp_model.model_shards):
            params = sum(w.shape.num_elements() for w in shard.weights)
            print(f"   Shard {i}: {params:,} parameters")
        
        # Test forward pass
        test_input = np.random.random((32, 100)).astype(np.float32)
        output = tp_model(test_input)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
        
        # Calculate efficiency
        total_shard_params = sum(sum(w.shape.num_elements() for w in shard.weights) for shard in tp_model.model_shards)
        efficiency = total_shard_params / total_params
        print(f"📈 Sharding efficiency: {efficiency:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ {strategy.upper()} strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def explain_sharding_strategies():
    """Explain the different sharding strategies."""
    print("🔍 Sharding Strategy Explanation:")
    print("=" * 50)
    
    print("\n📊 ROW Strategy:")
    print("   - Splits input features (dim=0)")
    print("   - Each shard gets ALL output features")
    print("   - Each shard can produce complete output independently")
    print("   - Good for: Reducing input processing per device")
    print("   - ⚠️  NOT IMPLEMENTED YET - requires complex input preprocessing")
    
    print("\n📊 COLUMN Strategy:")
    print("   - Splits output features (dim=1)")
    print("   - Each shard gets ALL input features")
    print("   - Each shard produces partial output")
    print("   - Good for: Reducing output size per device")
    print("   - ✅ FULLY IMPLEMENTED - working perfectly")
    
    print("\n📊 MIXED Strategy:")
    print("   - Alternates between row and column for different layers")
    print("   - Even layers: Row-wise sharding")
    print("   - Odd layers: Column-wise sharding")
    print("   - Good for: Balanced distribution across layers")
    print("   - ⚠️  NOT IMPLEMENTED YET - falls back to column-wise")
    
    print("\n📊 AUTO Strategy:")
    print("   - Default strategy (currently column-wise for Dense layers)")
    print("   - Automatically chooses best approach per layer type")
    print("   - Good for: Most use cases")
    print("   - ✅ FULLY IMPLEMENTED - uses column-wise for Dense layers")

if __name__ == "__main__":
    print("🚀 Testing Different Sharding Strategies...")
    
    # Explain strategies
    explain_sharding_strategies()
    
    # Test each strategy
    strategies = ["auto", "row", "column", "mixed"]
    results = {}
    
    for strategy in strategies:
        results[strategy] = test_sharding_strategy(strategy)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SHARDING STRATEGY TEST RESULTS:")
    print("=" * 50)
    
    for strategy, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{strategy.upper():<10}: {status}")
    
    successful_strategies = [s for s, r in results.items() if r]
    if successful_strategies:
        print(f"\n🎉 Successful strategies: {', '.join(successful_strategies)}")
    else:
        print("\n❌ All strategies failed!") 