#!/usr/bin/env python3
"""
Setup script for MDS dataset support in RWKV-X SFT training.
This script helps install dependencies and test MDS dataset loading.
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install required dependencies for MDS dataset support."""
    print("Installing dependencies for MDS dataset support...")

    try:
        # Install mosaicml-streaming
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "mosaicml-streaming>=0.6.0"
        ])
        print("✓ Successfully installed mosaicml-streaming")

        # Install other dependencies if not already installed
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Successfully installed all dependencies")

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

    return True


def test_mds_import():
    """Test if MDS dataset can be imported and used."""
    print("\nTesting MDS dataset import...")

    try:
        from streaming import StreamingDataset
        print(
            "✓ Successfully imported StreamingDataset from mosaicml-streaming")
        return True
    except ImportError as e:
        print(f"✗ Failed to import StreamingDataset: {e}")
        return False


def test_mds_dataset(data_path):
    """Test loading an MDS dataset from the given path."""
    print(f"\nTesting MDS dataset loading from: {data_path}")

    if not os.path.exists(data_path):
        print(f"✗ Data path does not exist: {data_path}")
        return False

    if not os.path.isdir(data_path):
        print(f"✗ Data path is not a directory: {data_path}")
        return False

    try:
        from streaming import StreamingDataset

        # Try to load the dataset
        dataset = StreamingDataset(
            local=data_path,
            split=None,
            shuffle=False,
            shuffle_algo="py1s",
            shuffle_seed=42,
            num_canonical_nodes=None,
            batch_size=None,
        )

        # Get dataset size
        dataset_size = len(dataset)
        print(f"✓ Successfully loaded MDS dataset with {dataset_size} samples")

        # Try to get a sample
        if dataset_size > 0:
            sample = dataset[0]
            print(f"✓ Successfully retrieved first sample")

            # Check sample structure
            if 'input_ids' in sample:
                input_ids = sample['input_ids']
                print(
                    f"✓ Sample contains input_ids with shape: {input_ids.shape if hasattr(input_ids, 'shape') else len(input_ids)}"
                )

            if 'domain' in sample:
                print(
                    f"✓ Sample contains domain information: {sample['domain']}"
                )

            if 'indices' in sample:
                print(f"✓ Sample contains indices information")

        return True

    except Exception as e:
        print(f"✗ Failed to load MDS dataset: {e}")
        return False


def main():
    """Main function to setup MDS dataset support."""
    print("RWKV-X MDS Dataset Setup")
    print("=" * 40)

    # # Install dependencies
    # if not install_dependencies():
    #     print("\nSetup failed. Please install dependencies manually.")
    #     return

    # # Test import
    # if not test_mds_import():
    #     print("\nSetup failed. Please check your installation.")
    #     return

    # Test dataset loading if path provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        if not test_mds_dataset(data_path):
            print(f"\nFailed to load dataset from {data_path}")
            return
    else:
        print("\nNo dataset path provided. To test dataset loading, run:")
        print(f"python {sys.argv[0]} /path/to/your/mds/dataset")

    print("\n✓ MDS dataset support setup completed successfully!")
    print("\nYou can now use MDS datasets with RWKV-X SFT training by:")
    print("1. Setting --data_file to your MDS dataset directory")
    print("2. Setting --data_type to 'mds' (or let it auto-detect)")
    print("3. Running the training script as usual")


if __name__ == "__main__":
    main()
