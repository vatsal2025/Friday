#!/usr/bin/env python3
"""
Step 7 Pipeline Execution Script
Runs the market data pipeline on real CSV data and captures execution metadata.
"""

import time
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_pipeline_command():
    """Run the market data pipeline CLI command and capture metadata."""
    
    print("=" * 80)
    print("STEP 7: Running Real CSV Data Through the Pipeline")
    print("=" * 80)
    
    # Find all prepared CSV files
    raw_dir = Path("data/raw")
    input_files = list(raw_dir.glob("prepared_*.csv"))
    
    if not input_files:
        raise FileNotFoundError("No prepared CSV files found in data/raw/")
    
    print(f"Found {len(input_files)} input files:")
    for file in input_files:
        print(f"  - {file}")
    print()
    
    # Process all files for a complete ~100MB dataset
    print("Processing all files for complete dataset...")
    
    # Track execution metadata
    start_time = time.time()
    start_datetime = datetime.now()
    
    # Initialize aggregated metadata
    metadata = {
        "execution_timestamp": start_datetime.isoformat(),
        "status": "started",
        "start_time": start_time,
        "duration_seconds": None,
        "total_files": len(input_files),
        "processed_files": 0,
        "failed_files": 0,
        "total_rows_processed": 0,
        "warnings": [],
        "errors": [],
        "file_results": [],
        "output_summary": {}
    }
    
    try:
        # Process each file individually
        for i, input_file in enumerate(input_files, 1):
            print(f"\nProcessing file {i}/{len(input_files)}: {input_file.name}")
            
            # Prepare the CLI command for this file
            cmd = [
                sys.executable, "-m", "pipelines.market_data_pipeline",
                "--input", str(input_file),
                "--outdir", "storage/data/processed", 
                "--warn-only",  # Allow validation warnings without stopping
                "--verbose"
            ]
            
            print(f"  Command: {' '.join(cmd)}")
            
            # Execute the command for this file
            file_start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="."
            )
            file_duration = time.time() - file_start_time
            
            # Track results for this file
            file_result = {
                "file_name": input_file.name,
                "file_path": str(input_file),
                "duration_seconds": round(file_duration, 2),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "rows_processed": 0
            }
            
            # Parse output for this file
            if result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "rows" in line.lower() and "processed" in line.lower():
                        words = line.split()
                        for word in words:
                            if word.replace(',', '').isdigit():
                                file_result["rows_processed"] = int(word.replace(',', ''))
                                metadata["total_rows_processed"] += int(word.replace(',', ''))
                                break
                    
                    if "warning" in line.lower():
                        metadata["warnings"].append(f"[{input_file.name}] {line.strip()}")
                    
                    if "error" in line.lower() and "successfully" not in line.lower():
                        metadata["errors"].append(f"[{input_file.name}] {line.strip()}")
            
            if result.stderr:
                # Add stderr content as warnings/errors
                stderr_lines = result.stderr.split('\n')
                for line in stderr_lines:
                    if line.strip():
                        if "warning" in line.lower() or "warn-only" in line.lower():
                            metadata["warnings"].append(f"[{input_file.name}] {line.strip()}")
                        else:
                            metadata["errors"].append(f"[{input_file.name}] {line.strip()}")
            
            # Update counters
            if result.returncode == 0:
                metadata["processed_files"] += 1
                print(f"  ✓ Successfully processed in {file_duration:.2f}s")
            else:
                metadata["failed_files"] += 1
                print(f"  ✗ Failed with return code {result.returncode}")
            
            metadata["file_results"].append(file_result)
        
        # Calculate total execution time
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Update final metadata
        metadata.update({
            "status": "completed" if metadata["failed_files"] == 0 else "partial_failure",
            "duration_seconds": round(total_duration, 2)
        })
        
        # Print overall results
        print(f"\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETED")
        print("=" * 60)
        print(f"Total files: {metadata['total_files']}")
        print(f"Successfully processed: {metadata['processed_files']}")
        print(f"Failed: {metadata['failed_files']}")
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Total rows processed: {metadata['total_rows_processed']:,}")
        
        # Check if output files were created
        output_dir = Path("storage/data/processed")
        if output_dir.exists():
            output_files = list(output_dir.rglob("*.parquet"))
            metadata["output_summary"] = {
                "output_directory": str(output_dir),
                "output_files_count": len(output_files),
                "output_files": [str(f) for f in output_files]
            }
        
        # Check for pipeline summary file
        summary_file = output_dir / "pipeline_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    pipeline_summary = json.load(f)
                metadata["pipeline_summary"] = pipeline_summary
            except Exception as e:
                metadata["warnings"].append(f"Could not read pipeline summary: {str(e)}")
        
        return metadata
        
    except Exception as e:
        end_time = time.time()
        total_duration = end_time - start_time
        
        metadata.update({
            "status": "error",
            "duration_seconds": round(total_duration, 2),
            "error_message": str(e),
            "errors": [str(e)]
        })
        
        print(f"Pipeline execution failed: {str(e)}")
        return metadata

def save_execution_metadata(metadata):
    """Save execution metadata to a JSON file."""
    
    output_file = "step7_execution_metadata.json"
    
    try:
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Execution metadata saved to: {output_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Status: {metadata['status']}")
        print(f"Duration: {metadata['duration_seconds']} seconds")
        print(f"Total rows processed: {metadata['total_rows_processed']:,}")
        print(f"Number of warnings: {len(metadata['warnings'])}")
        print(f"Number of errors: {len(metadata['errors'])}")
        
        if metadata.get('output_summary'):
            print(f"Output files created: {metadata['output_summary']['output_files_count']}")
            print(f"Output directory: {metadata['output_summary']['output_directory']}")
        
        if metadata['warnings']:
            print("\nWarnings:")
            for warning in metadata['warnings']:
                print(f"  - {warning}")
        
        if metadata['errors']:
            print("\nErrors:")
            for error in metadata['errors']:
                print(f"  - {error}")
        
        return True
        
    except Exception as e:
        print(f"Failed to save metadata: {str(e)}")
        return False

def main():
    """Main execution function."""
    
    # Ensure directories exist
    Path("storage/data/processed").mkdir(parents=True, exist_ok=True)
    
    # Run the pipeline
    metadata = run_pipeline_command()
    
    # Save metadata
    save_execution_metadata(metadata)
    
    # Return appropriate exit code
    if metadata['status'] == 'completed':
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
