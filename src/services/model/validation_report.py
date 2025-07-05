"""Validation Report Generator for Friday AI Trading System.

This module provides functionality for generating detailed validation reports
for model validation results, including metrics, comparisons, and visualizations.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine.model_versioning import ValidationResult
from src.services.model.validation_pipeline import ValidationPipelineResult
from src.services.model.benchmark_validation import BenchmarkResult, BenchmarkMetric

# Create logger
logger = get_logger(__name__)


class ReportFormat(Enum):
    """Format options for validation reports."""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"  # Requires additional dependencies


class ValidationReportGenerator:
    """Generator for model validation reports."""
    
    def __init__(self, output_dir: str = None):
        """Initialize a validation report generator.
        
        Args:
            output_dir: Directory to save reports to (defaults to current directory)
        """
        self.output_dir = output_dir or os.getcwd()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        logger.info(f"Initialized validation report generator with output directory: {self.output_dir}")
    
    def generate_report(self, model_name: str, model_version: str, 
                       pipeline_result: ValidationPipelineResult,
                       benchmark_results: List[BenchmarkResult] = None,
                       additional_metadata: Dict[str, Any] = None,
                       format: ReportFormat = ReportFormat.HTML,
                       include_visualizations: bool = True) -> str:
        """Generate a validation report.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            pipeline_result: Result of the validation pipeline
            benchmark_results: Results of benchmark validations
            additional_metadata: Additional metadata to include in the report
            format: Format of the report
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            str: Path to the generated report file
        """
        # Create report data
        report_data = self._create_report_data(
            model_name, model_version, pipeline_result, 
            benchmark_results, additional_metadata
        )
        
        # Generate report in the specified format
        if format == ReportFormat.JSON:
            report_path = self._generate_json_report(report_data, model_name, model_version)
        elif format == ReportFormat.HTML:
            report_path = self._generate_html_report(
                report_data, model_name, model_version, include_visualizations
            )
        elif format == ReportFormat.MARKDOWN:
            report_path = self._generate_markdown_report(
                report_data, model_name, model_version, include_visualizations
            )
        elif format == ReportFormat.PDF:
            report_path = self._generate_pdf_report(
                report_data, model_name, model_version, include_visualizations
            )
        else:
            raise ValueError(f"Unsupported report format: {format}")
        
        logger.info(f"Generated validation report: {report_path}")
        return report_path
    
    def _create_report_data(self, model_name: str, model_version: str,
                          pipeline_result: ValidationPipelineResult,
                          benchmark_results: List[BenchmarkResult] = None,
                          additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create the data for a validation report.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            pipeline_result: Result of the validation pipeline
            benchmark_results: Results of benchmark validations
            additional_metadata: Additional metadata to include in the report
            
        Returns:
            Dict[str, Any]: Report data
        """
        # Basic report data
        report_data = {
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat(),
            "validation_result": pipeline_result.overall_result.value,
            "validation_duration": pipeline_result.duration,
            "pipeline_results": pipeline_result.to_dict(),
            "metadata": additional_metadata or {}
        }
        
        # Add benchmark results if provided
        if benchmark_results:
            report_data["benchmark_results"] = [
                benchmark_result.to_dict() for benchmark_result in benchmark_results
            ]
        
        # Add summary statistics
        report_data["summary"] = {
            "total_rules": len(pipeline_result.rule_results),
            "passed_rules": len([r for r in pipeline_result.rule_results.values() 
                               if r["result"] == ValidationResult.PASS.value]),
            "warning_rules": len(pipeline_result.get_warnings()),
            "error_rules": len(pipeline_result.get_errors()),
            "failed_rules": len(pipeline_result.get_failed_rules())
        }
        
        return report_data
    
    def _generate_json_report(self, report_data: Dict[str, Any], 
                            model_name: str, model_version: str) -> str:
        """Generate a JSON validation report.
        
        Args:
            report_data: Report data
            model_name: Name of the model
            model_version: Version of the model
            
        Returns:
            str: Path to the generated report file
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_v{model_version}_{timestamp}_validation_report.json"
        file_path = os.path.join(self.output_dir, filename)
        
        # Write report to file
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return file_path
    
    def _generate_html_report(self, report_data: Dict[str, Any], 
                            model_name: str, model_version: str,
                            include_visualizations: bool = True) -> str:
        """Generate an HTML validation report.
        
        Args:
            report_data: Report data
            model_name: Name of the model
            model_version: Version of the model
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            str: Path to the generated report file
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_v{model_version}_{timestamp}_validation_report.html"
        file_path = os.path.join(self.output_dir, filename)
        
        # Generate HTML content
        html_content = self._generate_html_content(report_data, include_visualizations)
        
        # Write report to file
        with open(file_path, 'w') as f:
            f.write(html_content)
        
        return file_path
    
    def _generate_html_content(self, report_data: Dict[str, Any],
                             include_visualizations: bool = True) -> str:
        """Generate HTML content for a validation report.
        
        Args:
            report_data: Report data
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            str: HTML content
        """
        # Start HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report: {report_data['model_name']} v{report_data['model_version']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                .summary-box {{ background-color: #f5f5f5; border-radius: 5px; padding: 15px; width: 18%; text-align: center; }}
                .pass {{ background-color: #dff0d8; }}
                .warning {{ background-color: #fcf8e3; }}
                .error {{ background-color: #f2dede; }}
                .fail {{ background-color: #f2dede; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .visualization {{ margin: 20px 0; }}
                .result-PASS {{ color: green; }}
                .result-WARNING {{ color: orange; }}
                .result-ERROR {{ color: red; }}
                .result-FAIL {{ color: darkred; }}
            </style>
        </head>
        <body>
            <h1>Validation Report: {report_data['model_name']} v{report_data['model_version']}</h1>
            <p><strong>Generated:</strong> {report_data['timestamp']}</p>
            <p><strong>Overall Result:</strong> <span class="result-{report_data['validation_result']}">{report_data['validation_result']}</span></p>
            <p><strong>Validation Duration:</strong> {report_data['validation_duration']:.2f} seconds</p>
            
            <h2>Summary</h2>
            <div class="summary">
                <div class="summary-box pass">
                    <h3>{report_data['summary']['passed_rules']}</h3>
                    <p>Passed Rules</p>
                </div>
                <div class="summary-box warning">
                    <h3>{report_data['summary']['warning_rules']}</h3>
                    <p>Warnings</p>
                </div>
                <div class="summary-box error">
                    <h3>{report_data['summary']['error_rules']}</h3>
                    <p>Errors</p>
                </div>
                <div class="summary-box fail">
                    <h3>{report_data['summary']['failed_rules']}</h3>
                    <p>Failed Rules</p>
                </div>
                <div class="summary-box">
                    <h3>{report_data['summary']['total_rules']}</h3>
                    <p>Total Rules</p>
                </div>
            </div>
        """
        
        # Add validation pipeline results
        html += """
            <h2>Validation Pipeline Results</h2>
        """
        
        # Add stage results
        for stage_name, stage_data in report_data['pipeline_results']['stage_results'].items():
            html += f"""
            <h3>Stage: {stage_name}</h3>
            <p><strong>Result:</strong> <span class="result-{stage_data['result']}">{stage_data['result']}</span></p>
            <table>
                <tr>
                    <th>Rule</th>
                    <th>Type</th>
                    <th>Result</th>
                    <th>Message</th>
                </tr>
            """
            
            for rule in stage_data['rules']:
                html += f"""
                <tr>
                    <td>{rule['rule_name']}</td>
                    <td>{rule['rule_type']}</td>
                    <td class="result-{rule['result']}">{rule['result']}</td>
                    <td>{rule['message']}</td>
                </tr>
                """
            
            html += """
            </table>
            """
        
        # Add benchmark results if available
        if 'benchmark_results' in report_data and report_data['benchmark_results']:
            html += """
            <h2>Benchmark Results</h2>
            """
            
            for benchmark_result in report_data['benchmark_results']:
                html += f"""
                <h3>Benchmark: {benchmark_result['dataset_name']}</h3>
                <p><strong>Model:</strong> {benchmark_result['model_name']}</p>
                <p><strong>Result:</strong> <span class="result-{benchmark_result['validation_result']}">{benchmark_result['validation_result']}</span></p>
                <p><strong>Message:</strong> {benchmark_result['message']}</p>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Threshold</th>
                        <th>Comparison</th>
                        <th>Passed</th>
                    </tr>
                """
                
                for metric_name, metric_data in benchmark_result['metrics'].items():
                    # Handle different metric data formats
                    if 'benchmark_value' in metric_data:
                        # Comparison with benchmark model
                        html += f"""
                        <tr>
                            <td>{metric_name}</td>
                            <td>{metric_data['value']:.4f}</td>
                            <td>{metric_data['benchmark_value']:.4f} (benchmark)</td>
                            <td>{metric_data['comparison']}</td>
                            <td>{"Yes" if metric_data['passed'] else "No"}</td>
                        </tr>
                        """
                    else:
                        # Comparison with threshold
                        html += f"""
                        <tr>
                            <td>{metric_name}</td>
                            <td>{metric_data['value']:.4f}</td>
                            <td>{metric_data['threshold'] if metric_data['threshold'] is not None else 'N/A'}</td>
                            <td>{metric_data['comparison'] if metric_data['comparison'] is not None else 'N/A'}</td>
                            <td>{"Yes" if metric_data['passed'] else "No"}</td>
                        </tr>
                        """
                
                html += """
                </table>
                """
                
                # Add visualizations if enabled
                if include_visualizations:
                    # Generate visualizations for this benchmark result
                    viz_path = self._generate_benchmark_visualizations(benchmark_result, report_data['model_name'], report_data['model_version'])
                    if viz_path:
                        # Add the visualization image to the HTML
                        html += f"""
                        <div class="visualization">
                            <h4>Metric Comparison</h4>
                            <img src="{os.path.basename(viz_path)}" alt="Benchmark Comparison" width="800">
                        </div>
                        """
        
        # Add metadata if available
        if report_data['metadata']:
            html += """
            <h2>Additional Metadata</h2>
            <table>
                <tr>
                    <th>Key</th>
                    <th>Value</th>
                </tr>
            """
            
            for key, value in report_data['metadata'].items():
                html += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
                """
            
            html += """
            </table>
            """
        
        # End HTML content
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_markdown_report(self, report_data: Dict[str, Any],
                                model_name: str, model_version: str,
                                include_visualizations: bool = True) -> str:
        """Generate a Markdown validation report.
        
        Args:
            report_data: Report data
            model_name: Name of the model
            model_version: Version of the model
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            str: Path to the generated report file
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_v{model_version}_{timestamp}_validation_report.md"
        file_path = os.path.join(self.output_dir, filename)
        
        # Start Markdown content
        markdown = f"""# Validation Report: {report_data['model_name']} v{report_data['model_version']}

**Generated:** {report_data['timestamp']}  
**Overall Result:** {report_data['validation_result']}  
**Validation Duration:** {report_data['validation_duration']:.2f} seconds  

## Summary

- **Passed Rules:** {report_data['summary']['passed_rules']}
- **Warnings:** {report_data['summary']['warning_rules']}
- **Errors:** {report_data['summary']['error_rules']}
- **Failed Rules:** {report_data['summary']['failed_rules']}
- **Total Rules:** {report_data['summary']['total_rules']}

## Validation Pipeline Results
"""
        
        # Add stage results
        for stage_name, stage_data in report_data['pipeline_results']['stage_results'].items():
            markdown += f"""### Stage: {stage_name}

**Result:** {stage_data['result']}

| Rule | Type | Result | Message |
|------|------|--------|--------|
"""
            
            for rule in stage_data['rules']:
                markdown += f"| {rule['rule_name']} | {rule['rule_type']} | {rule['result']} | {rule['message']} |\n"
            
            markdown += "\n"
        
        # Add benchmark results if available
        if 'benchmark_results' in report_data and report_data['benchmark_results']:
            markdown += "\n## Benchmark Results\n"
            
            for benchmark_result in report_data['benchmark_results']:
                markdown += f"""### Benchmark: {benchmark_result['dataset_name']}

**Model:** {benchmark_result['model_name']}  
**Result:** {benchmark_result['validation_result']}  
**Message:** {benchmark_result['message']}  

| Metric | Value | Threshold | Comparison | Passed |
|--------|-------|-----------|------------|--------|
"""
                
                for metric_name, metric_data in benchmark_result['metrics'].items():
                    # Handle different metric data formats
                    if 'benchmark_value' in metric_data:
                        # Comparison with benchmark model
                        markdown += f"| {metric_name} | {metric_data['value']:.4f} | {metric_data['benchmark_value']:.4f} (benchmark) | {metric_data['comparison']} | {"Yes" if metric_data['passed'] else "No"} |\n"
                    else:
                        # Comparison with threshold
                        threshold = metric_data['threshold'] if metric_data['threshold'] is not None else 'N/A'
                        comparison = metric_data['comparison'] if metric_data['comparison'] is not None else 'N/A'
                        markdown += f"| {metric_name} | {metric_data['value']:.4f} | {threshold} | {comparison} | {"Yes" if metric_data['passed'] else "No"} |\n"
                
                markdown += "\n"
                
                # Add visualizations if enabled
                if include_visualizations:
                    # Generate visualizations for this benchmark result
                    viz_path = self._generate_benchmark_visualizations(benchmark_result, report_data['model_name'], report_data['model_version'])
                    if viz_path:
                        # Add the visualization image to the Markdown
                        markdown += f"""#### Metric Comparison

![Benchmark Comparison]({os.path.basename(viz_path)})\n\n"""
        
        # Add metadata if available
        if report_data['metadata']:
            markdown += "\n## Additional Metadata\n\n"
            markdown += "| Key | Value |\n|-----|-------|"
            
            for key, value in report_data['metadata'].items():
                markdown += f"\n| {key} | {value} |"
        
        # Write report to file
        with open(file_path, 'w') as f:
            f.write(markdown)
        
        return file_path
    
    def _generate_pdf_report(self, report_data: Dict[str, Any],
                           model_name: str, model_version: str,
                           include_visualizations: bool = True) -> str:
        """Generate a PDF validation report.
        
        Args:
            report_data: Report data
            model_name: Name of the model
            model_version: Version of the model
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            str: Path to the generated report file
        """
        try:
            import weasyprint
        except ImportError:
            logger.warning("weasyprint not installed. Falling back to HTML report.")
            return self._generate_html_report(report_data, model_name, model_version, include_visualizations)
        
        # First generate HTML report
        html_path = self._generate_html_report(report_data, model_name, model_version, include_visualizations)
        
        # Create PDF filename
        pdf_path = html_path.replace('.html', '.pdf')
        
        # Convert HTML to PDF
        try:
            with open(html_path, 'r') as f:
                html_content = f.read()
            
            weasyprint.HTML(string=html_content).write_pdf(pdf_path)
            logger.info(f"Generated PDF report: {pdf_path}")
            return pdf_path
        
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            logger.warning("Falling back to HTML report.")
            return html_path
    
    def _generate_benchmark_visualizations(self, benchmark_result: Dict[str, Any],
                                         model_name: str, model_version: str) -> Optional[str]:
        """Generate visualizations for benchmark results.
        
        Args:
            benchmark_result: Benchmark result data
            model_name: Name of the model
            model_version: Version of the model
            
        Returns:
            Optional[str]: Path to the generated visualization file, or None if generation failed
        """
        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_v{model_version}_{benchmark_result['dataset_name']}_{timestamp}_viz.png"
            file_path = os.path.join(self.output_dir, filename)
            
            # Extract metric data
            metrics = []
            values = []
            benchmark_values = []
            passed = []
            
            for metric_name, metric_data in benchmark_result['metrics'].items():
                metrics.append(metric_name)
                values.append(metric_data['value'])
                
                if 'benchmark_value' in metric_data:
                    benchmark_values.append(metric_data['benchmark_value'])
                else:
                    benchmark_values.append(None)
                
                passed.append(metric_data['passed'])
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Create bar chart for metrics
            x = np.arange(len(metrics))
            width = 0.35
            
            # Plot model values
            bars1 = plt.bar(x - width/2, values, width, label=f'{model_name} v{model_version}')
            
            # Plot benchmark values if available
            if any(v is not None for v in benchmark_values):
                # Filter out None values
                valid_indices = [i for i, v in enumerate(benchmark_values) if v is not None]
                valid_x = [x[i] for i in valid_indices]
                valid_benchmark_values = [benchmark_values[i] for i in valid_indices]
                
                bars2 = plt.bar(np.array(valid_x) + width/2, valid_benchmark_values, width, label='Benchmark')
            
            # Add labels and title
            plt.xlabel('Metrics')
            plt.ylabel('Values')
            plt.title(f'Benchmark Comparison: {benchmark_result["dataset_name"]}')
            plt.xticks(x, metrics, rotation=45, ha='right')
            plt.legend()
            
            # Add color coding for passed/failed metrics
            for i, (bar, p) in enumerate(zip(bars1, passed)):
                bar.set_color('green' if p else 'red')
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(file_path)
            plt.close()
            
            return file_path
        
        except Exception as e:
            logger.error(f"Error generating benchmark visualizations: {str(e)}")
            return None


# Create a singleton instance
validation_report_generator = ValidationReportGenerator()