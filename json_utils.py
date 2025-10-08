import json
import numpy as np
from pathlib import Path

def convert_to_json_serializable(obj):
    """
    Convert numpy types and other non-JSON-serializable types to JSON-compatible types
    """
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def safe_json_save(data, filepath):
    """
    Safely save data to JSON file, handling numpy types
    """
    try:
        # Convert data to JSON-serializable format
        serializable_data = convert_to_json_serializable(data)
        
        # Save to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"✅ Data saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save JSON to {filepath}: {e}")
        return False

def safe_json_load(filepath):
    """
    Safely load JSON data from file
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Data loaded from {filepath}")
        return data
        
    except Exception as e:
        print(f"❌ Failed to load JSON from {filepath}: {e}")
        return None

# Test the utility functions
if __name__ == "__main__":
    # Test with numpy types
    test_data = {
        'float32_val': np.float32(3.14159),
        'int64_val': np.int64(42),
        'array': np.array([1, 2, 3, 4]),
        'nested_dict': {
            'loss': np.float32(0.5),
            'epoch': np.int32(10)
        },
        'list_with_numpy': [np.float32(1.1), np.float32(2.2)]
    }
    
    print("🧪 Testing JSON serialization utilities...")
    
    # Test save
    test_file = "test_output.json"
    success = safe_json_save(test_data, test_file)
    
    if success:
        # Test load
        loaded_data = safe_json_load(test_file)
        
        if loaded_data:
            print("✅ JSON utilities working correctly!")
            
            # Clean up test file
            Path(test_file).unlink()
        else:
            print("❌ Failed to load test data")
    else:
        print("❌ Failed to save test data")