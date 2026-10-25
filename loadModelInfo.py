import torch
import sys
from collections import OrderedDict


def inspect_pth_values(pth_path):
    try:
        # Load the .pth file
        checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))

        # Verify that the loaded object is a dictionary
        if not isinstance(checkpoint, dict):
            print("Error: Loaded object is not a dictionary.")
            sys.exit(1)

        # Define the expected keys
        expected_keys = [
            'run_number',
            'model_state_dict',
            'calibrated_params',
            'final_metrics',
            'best_epoch',
            'confusion_matrix',
            'training_history'
        ]

        # Check if all expected keys are present
        missing_keys = [key for key in expected_keys if key not in checkpoint]
        if missing_keys:
            print(f"Warning: The following expected keys are missing: {missing_keys}")

        # Iterate through each key and print its value
        print(f"\nInspecting values in checkpoint: {pth_path}\n")
        for key in expected_keys:
            if key in checkpoint:
                print(f"--- {key} ---")
                value = checkpoint[key]

                # Handle different types of values
                if key == 'model_state_dict' and isinstance(value, (dict, OrderedDict)):
                    print("Type: State Dictionary")
                    print("Keys in model_state_dict (model parameters):")
                    for param_name in value.keys():
                        print(f"  - {param_name}")
                        # Optionally print shape of tensor
                        if isinstance(value[param_name], torch.Tensor):
                            print(f"    Shape: {list(value[param_name].shape)}")
                elif isinstance(value, (dict, OrderedDict)):
                    print("Type: Dictionary")
                    print("Contents:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                elif isinstance(value, (list, tuple)):
                    print("Type: List/Tuple")
                    print(f"Length: {len(value)}")
                    print("Contents:", value)
                elif isinstance(value, torch.Tensor):
                    print("Type: Tensor")
                    print(f"Shape: {list(value.shape)}")
                    print(f"Value:\n{value}")
                else:
                    print(f"Type: {type(value).__name__}")
                    print(f"Value: {value}")
                print()  # Add newline for readability
            else:
                print(f"Key '{key}' not found in checkpoint.\n")

    except Exception as e:
        print(f"Error loading .pth file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    pth_path = "BestModels/PaviaHOS7738_run066.pth"  # Path to your .pth file
    inspect_pth_values(pth_path)