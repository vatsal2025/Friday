"""Validation report generator for model validation framework.

This module provides functionality to generate detailed validation reports
in various formats (JSON, HTML, Markdown, PDF) with comprehensive metrics.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

from src.services.model.validation_pipeline import ValidationPipelineResult, ValidationStage
from src.services.model.model_validation_rules import ValidationResult, ModelType

logger = logging.getLogger(__name__)


class ValidationReportGenerator:
    """Generator for comprehensive model validation reports."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the validation report generator.
        
        Args:
            output_dir: Directory to save reports (defaults to current directory)
        """
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(
        self,
        pipeline_result: Union[ValidationPipelineResult, Dict[str, Any]],
        benchmark_results: Optional[List[Dict[str, Any]]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
        format: str = "json",
        filename: Optional[str] = None
    ) -> str:
        """Generate a validation report in the specified format.
        
        Args:
            pipeline_result: The validation pipeline result
            benchmark_results: Optional list of benchmark validation results
            model_metadata: Optional model metadata
            format: Report format ("json", "html", "markdown", or "pdf")
            filename: Optional filename for the report
            
        Returns:
            Path to the generated report file
        """
        # Convert pipeline_result to ValidationPipelineResult if it's a dict
        if isinstance(pipeline_result, dict):
            pipeline_result = ValidationPipelineResult.from_dict(pipeline_result)
        
        # Generate report data
        report_data = self._prepare_report_data(pipeline_result, benchmark_results, model_metadata)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = model_metadata.get("name", "model") if model_metadata else "model"
            filename = f"{model_name}_validation_report_{timestamp}"
        
        # Generate report in specified format
        if format.lower() == "json":
            return self._generate_json_report(report_data, filename)
        elif format.lower() == "html":
            return self._generate_html_report(report_data, filename)
        elif format.lower() == "markdown" or format.lower() == "md":
            return self._generate_markdown_report(report_data, filename)
        elif format.lower() == "pdf":
            return self._generate_pdf_report(report_data, filename)
        else:
            logger.warning(f"Unsupported report format: {format}. Defaulting to JSON.")
            return self._generate_json_report(report_data, filename)
    
    def _prepare_report_data(self, 
                            pipeline_result: ValidationPipelineResult,
                            benchmark_results: Optional[List[Dict[str, Any]]] = None,
                            model_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare data for the validation report.
        
        Args:
            pipeline_result: The validation pipeline result
            benchmark_results: Optional list of benchmark validation results
            model_metadata: Optional model metadata
            
        Returns:
            Dictionary containing the report data
        """
        # Basic report data
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_result": pipeline_result.overall_result.name,
            "pipeline_name": pipeline_result.pipeline_name,
            "duration_seconds": pipeline_result.duration_seconds,
            "model_metadata": model_metadata or {},
            "stages": {},
            "summary": {
                "total_rules": len(pipeline_result.rule_results),
                "passed": len([r for r in pipeline_result.rule_results if r.result == ValidationResult.PASS]),
                "warnings": len([r for r in pipeline_result.rule_results if r.result == ValidationResult.WARNING]),
                "failed": len([r for r in pipeline_result.rule_results if r.result == ValidationResult.FAIL]),
                "errors": len([r for r in pipeline_result.rule_results if r.result == ValidationResult.ERROR]),
            }
        }
        
        # Add stage results
        for stage in ValidationStage:
            stage_results = [r for r in pipeline_result.rule_results if r.stage == stage]
            if stage_results:
                report_data["stages"][stage.name] = {
                    "result": pipeline_result.stage_results.get(stage, ValidationResult.PASS).name,
                    "rules": [
                        {
                            "name": r.rule_name,
                            "result": r.result.name,
                            "message": r.message,
                            "duration_seconds": r.duration_seconds
                        } for r in stage_results
                    ]
                }
        
        # Add benchmark results if provided
        if benchmark_results:
            report_data["benchmark_results"] = benchmark_results
        
        return report_data
    
    def _generate_json_report(self, report_data: Dict[str, Any], filename: str) -> str:
        """Generate a JSON validation report.
        
        Args:
            report_data: The report data
            filename: The filename for the report
            
        Returns:
            Path to the generated report file
        """
        if not filename.endswith(".json"):
            filename += ".json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, "w") as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Generated JSON validation report: {file_path}")
        return file_path
    
    def _generate_html_report(self, report_data: Dict[str, Any], filename: str) -> str:
        """Generate an HTML validation report.
        
        Args:
            report_data: The report data
            filename: The filename for the report
            
        Returns:
            Path to the generated report file
        """
        if not filename.endswith(".html"):
            filename += ".html"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # HTML template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .summary { margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
                .pass { color: green; }
                .warning { color: orange; }
                .fail { color: red; }
                .error { color: darkred; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .stage { margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .benchmark { margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .metadata { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Model Validation Report</h1>
            <p><strong>Generated:</strong> {{ report_data.timestamp }}</p>
            <p><strong>Pipeline:</strong> {{ report_data.pipeline_name }}</p>
            <p><strong>Duration:</strong> {{ report_data.duration_seconds }} seconds</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Overall Result:</strong> <span class="{{ report_data.overall_result.lower() }}">{{ report_data.overall_result }}</span></p>
                <p><strong>Total Rules:</strong> {{ report_data.summary.total_rules }}</p>
                <p><strong>Passed:</strong> <span class="pass">{{ report_data.summary.passed }}</span></p>
                <p><strong>Warnings:</strong> <span class="warning">{{ report_data.summary.warnings }}</span></p>
                <p><strong>Failed:</strong> <span class="fail">{{ report_data.summary.failed }}</span></p>
                <p><strong>Errors:</strong> <span class="error">{{ report_data.summary.errors }}</span></p>
            </div>
            
            {% if report_data.model_metadata %}
            <div class="metadata">
                <h2>Model Metadata</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    {% for key, value in report_data.model_metadata.items() %}
                    <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            <h2>Validation Stages</h2>
            {% for stage_name, stage_data in report_data.stages.items() %}
            <div class="stage">
                <h3>{{ stage_name }} Stage</h3>
                <p><strong>Result:</strong> <span class="{{ stage_data.result.lower() }}">{{ stage_data.result }}</span></p>
                
                <table>
                    <tr>
                        <th>Rule</th>
                        <th>Result</th>
                        <th>Message</th>
                        <th>Duration (s)</th>
                    </tr>
                    {% for rule in stage_data.rules %}
                    <tr>
                        <td>{{ rule.name }}</td>
                        <td class="{{ rule.result.lower() }}">{{ rule.result }}</td>
                        <td>{{ rule.message }}</td>
                        <td>{{ rule.duration_seconds }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endfor %}
            
            {% if report_data.benchmark_results %}
            <h2>Benchmark Results</h2>
            {% for benchmark in report_data.benchmark_results %}
            <div class="benchmark">
                <h3>{{ benchmark.dataset_name }}</h3>
                <p><strong>Result:</strong> <span class="{{ benchmark.result.lower() }}">{{ benchmark.result }}</span></p>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Benchmark Value</th>
                        <th>Improvement</th>
                        <th>Required Improvement</th>
                    </tr>
                    {% for metric in benchmark.metrics %}
                    <tr>
                        <td>{{ metric.name }}</td>
                        <td>{{ metric.value }}</td>
                        <td>{{ metric.benchmark_value }}</td>
                        <td>{{ metric.improvement }}</td>
                        <td>{{ metric.required_improvement }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endfor %}
            {% endif %}
        </body>
        </html>
        """
        
        # Render template
        template = Template(template_str)
        html_content = template.render(report_data=report_data)
        
        with open(file_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML validation report: {file_path}")
        return file_path
    
    def _generate_markdown_report(self, report_data: Dict[str, Any], filename: str) -> str:
        """Generate a Markdown validation report.
        
        Args:
            report_data: The report data
            filename: The filename for the report
            
        Returns:
            Path to the generated report file
        """
        if not filename.endswith(".md"):
            filename += ".md"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Build markdown content
        md_content = []
        
        # Header
        md_content.append("# Model Validation Report\n")
        md_content.append(f"**Generated:** {report_data['timestamp']}\n")
        md_content.append(f"**Pipeline:** {report_data['pipeline_name']}\n")
        md_content.append(f"**Duration:** {report_data['duration_seconds']} seconds\n")
        
        # Summary
        md_content.append("## Summary\n")
        md_content.append(f"**Overall Result:** {report_data['overall_result']}\n")
        md_content.append(f"**Total Rules:** {report_data['summary']['total_rules']}\n")
        md_content.append(f"**Passed:** {report_data['summary']['passed']}\n")
        md_content.append(f"**Warnings:** {report_data['summary']['warnings']}\n")
        md_content.append(f"**Failed:** {report_data['summary']['failed']}\n")
        md_content.append(f"**Errors:** {report_data['summary']['errors']}\n")
        
        # Model Metadata
        if report_data.get("model_metadata"):
            md_content.append("## Model Metadata\n")
            md_content.append("| Property | Value |\n")
            md_content.append("| --- | --- |\n")
            for key, value in report_data["model_metadata"].items():
                md_content.append(f"| {key} | {value} |\n")
            md_content.append("\n")
        
        # Validation Stages
        md_content.append("## Validation Stages\n")
        for stage_name, stage_data in report_data["stages"].items():
            md_content.append(f"### {stage_name} Stage\n")
            md_content.append(f"**Result:** {stage_data['result']}\n\n")
            
            md_content.append("| Rule | Result | Message | Duration (s) |\n")
            md_content.append("| --- | --- | --- | --- |\n")
            for rule in stage_data["rules"]:
                md_content.append(f"| {rule['name']} | {rule['result']} | {rule['message']} | {rule['duration_seconds']} |\n")
            md_content.append("\n")
        
        # Benchmark Results
        if report_data.get("benchmark_results"):
            md_content.append("## Benchmark Results\n")
            for benchmark in report_data["benchmark_results"]:
                md_content.append(f"### {benchmark['dataset_name']}\n")
                md_content.append(f"**Result:** {benchmark['result']}\n\n")
                
                md_content.append("| Metric | Value | Benchmark Value | Improvement | Required Improvement |\n")
                md_content.append("| --- | --- | --- | --- | --- |\n")
                for metric in benchmark["metrics"]:
                    md_content.append(f"| {metric['name']} | {metric['value']} | {metric['benchmark_value']} | {metric['improvement']} | {metric['required_improvement']} |\n")
                md_content.append("\n")
        
        # Write to file
        with open(file_path, "w") as f:
            f.write("".join(md_content))
        
        logger.info(f"Generated Markdown validation report: {file_path}")
        return file_path
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], filename: str) -> str:
        """Generate a PDF validation report.
        
        Args:
            report_data: The report data
            filename: The filename for the report
            
        Returns:
            Path to the generated report file
        """
        try:
            import weasyprint
        except ImportError:
            logger.warning("WeasyPrint not installed. Falling back to HTML report.")
            html_path = self._generate_html_report(report_data, filename)
            return html_path
        
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Generate HTML report first
        html_filename = f"{os.path.splitext(filename)[0]}_temp.html"
        html_path = self._generate_html_report(report_data, html_filename)
        
        # Convert HTML to PDF
        html = weasyprint.HTML(filename=html_path)
        html.write_pdf(file_path)
        
        # Remove temporary HTML file
        os.remove(html_path)
        
        logger.info(f"Generated PDF validation report: {file_path}")
        return file_path
    
    def generate_performance_charts(self, 
                                   pipeline_result: ValidationPipelineResult,
                                   benchmark_results: Optional[List[Dict[str, Any]]] = None,
                                   output_dir: Optional[str] = None) -> List[str]:
        """Generate performance charts for the validation report.
        
        Args:
            pipeline_result: The validation pipeline result
            benchmark_results: Optional list of benchmark validation results
            output_dir: Optional directory to save charts
            
        Returns:
            List of paths to the generated chart files
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        chart_paths = []
        
        # Set style
        sns.set_style("whitegrid")
        
        # Generate rule execution time chart
        if pipeline_result.rule_results:
            plt.figure(figsize=(10, 6))
            
            # Prepare data
            rule_names = [r.rule_name for r in pipeline_result.rule_results]
            durations = [r.duration_seconds for r in pipeline_result.rule_results]
            results = [r.result.name for r in pipeline_result.rule_results]
            
            # Create color map
            color_map = {
                "PASS": "green",
                "WARNING": "orange",
                "FAIL": "red",
                "ERROR": "darkred"
            }
            colors = [color_map.get(result, "blue") for result in results]
            
            # Create bar chart
            bars = plt.barh(rule_names, durations, color=colors)
            
            # Add labels and title
            plt.xlabel("Execution Time (seconds)")
            plt.ylabel("Validation Rule")
            plt.title("Rule Execution Times")
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = os.path.join(output_dir, f"rule_execution_times_{timestamp}.png")
            plt.savefig(chart_path)
            plt.close()
            
            chart_paths.append(chart_path)
        
        # Generate benchmark comparison chart if benchmark results are provided
        if benchmark_results:
            for benchmark in benchmark_results:
                if "metrics" in benchmark:
                    plt.figure(figsize=(10, 6))
                    
                    # Prepare data
                    metric_names = [m["name"] for m in benchmark["metrics"]]
                    model_values = [m["value"] for m in benchmark["metrics"]]
                    benchmark_values = [m["benchmark_value"] for m in benchmark["metrics"]]
                    
                    # Set up bar positions
                    x = range(len(metric_names))
                    width = 0.35
                    
                    # Create grouped bar chart
                    plt.bar([i - width/2 for i in x], model_values, width, label="Model")
                    plt.bar([i + width/2 for i in x], benchmark_values, width, label="Benchmark")
                    
                    # Add labels and title
                    plt.xlabel("Metric")
                    plt.ylabel("Value")
                    plt.title(f"Benchmark Comparison: {benchmark['dataset_name']}")
                    plt.xticks(x, metric_names)
                    plt.legend()
                    plt.tight_layout()
                    
                    # Save chart
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dataset_name = benchmark["dataset_name"].replace(" ", "_").lower()
                    chart_path = os.path.join(output_dir, f"benchmark_comparison_{dataset_name}_{timestamp}.png")
                    plt.savefig(chart_path)
                    plt.close()
                    
                    chart_paths.append(chart_path)
        
        return chart_paths


# Create a singleton instance
validation_report_generator = ValidationReportGenerator()