#!/usr/bin/env python3
# test_full_pipeline_automation.py

import os
import sys
import subprocess
import time

def main():
    """Test the full pipeline automation with integrated visualizations."""
    print("Testing full pipeline automation with integrated visualizations...")
    
    # Create a test directory
    test_dir = os.path.join(os.getcwd(), 'test_full_pipeline')
    os.makedirs(test_dir, exist_ok=True)
    
    # Path to the repository
    repo_dir = os.getcwd()
    
    # Create a test config file
    config_path = os.path.join(test_dir, 'test_config.yaml')
    with open(config_path, 'w') as f:
        f.write("""
output_dir: test_output
wandb_project: spatial-transcriptomics-test
max_epochs: 2
patience: 1
learning_rate: 0.001
batch_size: 16
        """)
    
    # Run the full pipeline
    print("\nRunning full pipeline with automatic visualizations...")
    cmd = [
        'python', 
        os.path.join(repo_dir, 'scripts', 'run_pipeline.py'),
        '--config', config_path,
        '--experiment', 'test_automation',
        '--step', 'all'
    ]
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=repo_dir
        )
        
        # Print output in real-time
        print("\nPipeline output:")
        print("-" * 50)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        # Check if visualization files were created
        vis_dir = os.path.join(test_dir, 'test_output', 'visualizations')
        if os.path.exists(vis_dir):
            vis_files = os.listdir(vis_dir)
            print("\nVisualization files created:")
            for file in vis_files:
                print(f"- {file}")
        
        # Check if the pipeline completed successfully
        if return_code == 0:
            print("\n✓ Full pipeline with automatic visualizations completed successfully!")
            return True
        else:
            print(f"\n✗ Pipeline failed with return code {return_code}")
            # Print stderr
            print("\nError output:")
            print("-" * 50)
            print(process.stderr.read())
            return False
    
    except Exception as e:
        print(f"\n✗ Error running pipeline: {e}")
        return False

if __name__ == "__main__":
    main()
