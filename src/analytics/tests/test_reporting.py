"""Tests for the analytics reporting module."""

import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import pandas as pd
import numpy as np

# Import reporting module
from analytics.reporting import (
    ReportFormat,
    VisualizationType,
    ReportTemplate,
    EnhancedReport,
    ReportingEngine
)


class TestReportFormat(unittest.TestCase):
    """Test cases for ReportFormat enum."""

    def test_report_format_values(self):
        """Test ReportFormat enum values."""
        self.assertEqual(ReportFormat.HTML.value, 'html')
        self.assertEqual(ReportFormat.PDF.value, 'pdf')
        self.assertEqual(ReportFormat.MARKDOWN.value, 'markdown')
        self.assertEqual(ReportFormat.JUPYTER.value, 'jupyter')
        self.assertEqual(ReportFormat.EXCEL.value, 'excel')


class TestVisualizationType(unittest.TestCase):
    """Test cases for VisualizationType enum."""

    def test_visualization_type_categories(self):
        """Test VisualizationType enum categories."""
        # Performance visualizations
        self.assertTrue(hasattr(VisualizationType, 'EQUITY_CURVE'))
        self.assertTrue(hasattr(VisualizationType, 'DRAWDOWN'))
        self.assertTrue(hasattr(VisualizationType, 'RETURNS_DISTRIBUTION'))
        self.assertTrue(hasattr(VisualizationType, 'MONTHLY_RETURNS'))
        
        # Allocation visualizations
        self.assertTrue(hasattr(VisualizationType, 'ASSET_ALLOCATION'))
        self.assertTrue(hasattr(VisualizationType, 'ALLOCATION_DRIFT'))
        self.assertTrue(hasattr(VisualizationType, 'SECTOR_ALLOCATION'))
        
        # Risk visualizations
        self.assertTrue(hasattr(VisualizationType, 'RISK_CONTRIBUTION'))
        self.assertTrue(hasattr(VisualizationType, 'CORRELATION_MATRIX'))
        self.assertTrue(hasattr(VisualizationType, 'FACTOR_EXPOSURES'))
        
        # Tax visualizations
        self.assertTrue(hasattr(VisualizationType, 'REALIZED_GAINS'))
        self.assertTrue(hasattr(VisualizationType, 'TAX_IMPACT'))
        
        # Comparative visualizations
        self.assertTrue(hasattr(VisualizationType, 'STRATEGY_COMPARISON'))
        self.assertTrue(hasattr(VisualizationType, 'SCENARIO_ANALYSIS'))


class TestReportTemplate(unittest.TestCase):
    """Test cases for ReportTemplate class."""

    def setUp(self):
        """Set up test fixtures."""
        self.template = ReportTemplate(
            name='Test Template',
            description='A test template',
            sections=[
                {
                    'title': 'Performance',
                    'visualizations': [
                        VisualizationType.EQUITY_CURVE,
                        VisualizationType.DRAWDOWN
                    ]
                },
                {
                    'title': 'Risk',
                    'visualizations': [
                        VisualizationType.RISK_CONTRIBUTION,
                        VisualizationType.CORRELATION_MATRIX
                    ]
                }
            ],
            metadata={
                'author': 'Test Author',
                'company': 'Test Company'
            }
        )

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.template.name, 'Test Template')
        self.assertEqual(self.template.description, 'A test template')
        self.assertEqual(len(self.template.sections), 2)
        self.assertEqual(self.template.metadata['author'], 'Test Author')

    def test_add_section(self):
        """Test adding a section."""
        self.template.add_section(
            'Allocation',
            [VisualizationType.ASSET_ALLOCATION, VisualizationType.ALLOCATION_DRIFT]
        )
        self.assertEqual(len(self.template.sections), 3)
        self.assertEqual(self.template.sections[2]['title'], 'Allocation')
        self.assertEqual(len(self.template.sections[2]['visualizations']), 2)

    def test_remove_section(self):
        """Test removing a section."""
        self.template.remove_section('Risk')
        self.assertEqual(len(self.template.sections), 1)
        self.assertEqual(self.template.sections[0]['title'], 'Performance')

    def test_update_metadata(self):
        """Test updating metadata."""
        self.template.update_metadata({
            'author': 'New Author',
            'date': '2023-01-01'
        })
        self.assertEqual(self.template.metadata['author'], 'New Author')
        self.assertEqual(self.template.metadata['date'], '2023-01-01')
        self.assertEqual(self.template.metadata['company'], 'Test Company')  # Unchanged

    def test_to_dict(self):
        """Test converting to dictionary."""
        template_dict = self.template.to_dict()
        self.assertEqual(template_dict['name'], 'Test Template')
        self.assertEqual(template_dict['description'], 'A test template')
        self.assertEqual(len(template_dict['sections']), 2)
        self.assertEqual(template_dict['metadata']['author'], 'Test Author')

    def test_from_dict(self):
        """Test creating from dictionary."""
        template_dict = {
            'name': 'New Template',
            'description': 'A new template',
            'sections': [
                {
                    'title': 'Section 1',
                    'visualizations': [
                        VisualizationType.EQUITY_CURVE.value,
                        VisualizationType.DRAWDOWN.value
                    ]
                }
            ],
            'metadata': {
                'author': 'New Author'
            }
        }
        template = ReportTemplate.from_dict(template_dict)
        self.assertEqual(template.name, 'New Template')
        self.assertEqual(template.description, 'A new template')
        self.assertEqual(len(template.sections), 1)
        self.assertEqual(template.metadata['author'], 'New Author')

    def test_save_and_load(self):
        """Test saving and loading template."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
            temp_name = temp.name

        try:
            # Save template
            self.template.save(temp_name)
            self.assertTrue(os.path.exists(temp_name))
            self.assertTrue(os.path.getsize(temp_name) > 0)

            # Load template
            loaded_template = ReportTemplate.load(temp_name)
            self.assertEqual(loaded_template.name, 'Test Template')
            self.assertEqual(loaded_template.description, 'A test template')
            self.assertEqual(len(loaded_template.sections), 2)
            self.assertEqual(loaded_template.metadata['author'], 'Test Author')
        finally:
            os.unlink(temp_name)


class TestEnhancedReport(unittest.TestCase):
    """Test cases for EnhancedReport class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.02, 100)
        cumulative_returns = (1 + returns).cumprod() - 1
        drawdowns = np.random.uniform(-0.3, 0, 100)
        drawdowns.sort()  # Sort to simulate realistic drawdowns

        self.returns = pd.Series(returns, index=dates)
        self.cumulative_returns = pd.Series(cumulative_returns, index=dates)
        self.drawdowns = pd.Series(drawdowns, index=dates)

        # Create benchmark data
        benchmark_returns = np.random.normal(0.0008, 0.018, 100)
        benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1
        self.benchmark_returns = pd.Series(benchmark_returns, index=dates)
        self.benchmark_cumulative_returns = pd.Series(benchmark_cumulative_returns, index=dates)

        # Create asset allocation data
        self.asset_classes = ['Equities', 'Bonds', 'Real Estate', 'Commodities', 'Cash']
        self.allocations = pd.Series([0.4, 0.3, 0.1, 0.1, 0.1], index=self.asset_classes)

        # Create template
        self.template = ReportTemplate(
            name='Test Template',
            description='A test template',
            sections=[
                {
                    'title': 'Performance',
                    'visualizations': [
                        VisualizationType.EQUITY_CURVE,
                        VisualizationType.DRAWDOWN
                    ]
                },
                {
                    'title': 'Allocation',
                    'visualizations': [
                        VisualizationType.ASSET_ALLOCATION
                    ]
                }
            ]
        )

        # Create report
        self.report = EnhancedReport(
            title='Test Report',
            portfolio_id='TEST123',
            start_date='2020-01-01',
            end_date='2020-04-10',
            template=self.template
        )

    @patch('analytics.reporting.PerformanceVisualizer')
    @patch('analytics.reporting.AllocationVisualizer')
    def test_generate_report(self, mock_allocation_visualizer, mock_performance_visualizer):
        """Test generating report."""
        # Mock visualizer methods
        mock_performance_visualizer.return_value.plot_equity_curve.return_value = 'equity_curve_fig'
        mock_performance_visualizer.return_value.plot_drawdown.return_value = 'drawdown_fig'
        mock_allocation_visualizer.return_value.plot_asset_allocation.return_value = 'asset_allocation_fig'

        # Generate report
        report_data = self.report.generate(
            returns=self.returns,
            cumulative_returns=self.cumulative_returns,
            drawdowns=self.drawdowns,
            benchmark_returns=self.benchmark_returns,
            benchmark_cumulative_returns=self.benchmark_cumulative_returns,
            allocations=self.allocations,
            format=ReportFormat.HTML
        )

        # Check report data
        self.assertEqual(report_data['title'], 'Test Report')
        self.assertEqual(report_data['portfolio_id'], 'TEST123')
        self.assertEqual(report_data['start_date'], '2020-01-01')
        self.assertEqual(report_data['end_date'], '2020-04-10')
        self.assertEqual(len(report_data['sections']), 2)
        self.assertEqual(report_data['format'], ReportFormat.HTML)

        # Check that visualizers were called
        mock_performance_visualizer.return_value.plot_equity_curve.assert_called_once()
        mock_performance_visualizer.return_value.plot_drawdown.assert_called_once()
        mock_allocation_visualizer.return_value.plot_asset_allocation.assert_called_once()

    @patch('analytics.reporting.PerformanceVisualizer')
    def test_calculate_metrics(self, mock_performance_visualizer):
        """Test calculating performance metrics."""
        # Mock metrics calculation
        mock_metrics = {
            'Total Return': 0.15,
            'Annualized Return': 0.12,
            'Sharpe Ratio': 1.2,
            'Sortino Ratio': 1.5,
            'Max Drawdown': -0.12,
            'Volatility': 0.10,
            'Alpha': 0.02,
            'Beta': 0.85
        }
        
        # Set up the mock
        self.report._calculate_performance_metrics = MagicMock(return_value=mock_metrics)
        
        # Calculate metrics
        metrics = self.report.calculate_metrics(
            returns=self.returns,
            benchmark_returns=self.benchmark_returns
        )
        
        # Check metrics
        self.assertEqual(metrics, mock_metrics)
        self.report._calculate_performance_metrics.assert_called_once()

    @patch('analytics.reporting.pd.DataFrame.to_html')
    @patch('analytics.reporting.pd.DataFrame.to_markdown')
    @patch('analytics.reporting.pd.DataFrame.to_excel')
    def test_export_metrics(self, mock_to_excel, mock_to_markdown, mock_to_html):
        """Test exporting metrics."""
        # Mock metrics
        metrics = {
            'Total Return': 0.15,
            'Annualized Return': 0.12,
            'Sharpe Ratio': 1.2,
            'Max Drawdown': -0.12
        }
        
        # Set up mocks
        mock_to_html.return_value = '<table>...</table>'
        mock_to_markdown.return_value = '| Metric | Value |\n|--------|-------|...'
        
        # Test HTML export
        result = self.report.export_metrics(metrics, ReportFormat.HTML)
        mock_to_html.assert_called_once()
        self.assertEqual(result, '<table>...</table>')
        
        # Test Markdown export
        result = self.report.export_metrics(metrics, ReportFormat.MARKDOWN)
        mock_to_markdown.assert_called_once()
        self.assertEqual(result, '| Metric | Value |\n|--------|-------|...')
        
        # Test Excel export
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp:
            temp_name = temp.name
            
        try:
            result = self.report.export_metrics(metrics, ReportFormat.EXCEL, temp_name)
            mock_to_excel.assert_called_once_with(temp_name)
            self.assertEqual(result, temp_name)
        finally:
            os.unlink(temp_name)


class TestReportingEngine(unittest.TestCase):
    """Test cases for ReportingEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = ReportingEngine()

    def test_init(self):
        """Test initialization."""
        self.assertEqual(len(self.engine.templates), 0)

    def test_register_template(self):
        """Test registering a template."""
        template = ReportTemplate(
            name='Test Template',
            description='A test template',
            sections=[
                {
                    'title': 'Performance',
                    'visualizations': [
                        VisualizationType.EQUITY_CURVE,
                        VisualizationType.DRAWDOWN
                    ]
                }
            ]
        )
        self.engine.register_template(template)
        self.assertEqual(len(self.engine.templates), 1)
        self.assertEqual(self.engine.templates['Test Template'], template)

    def test_get_template(self):
        """Test getting a template."""
        template = ReportTemplate(
            name='Test Template',
            description='A test template'
        )
        self.engine.register_template(template)
        retrieved_template = self.engine.get_template('Test Template')
        self.assertEqual(retrieved_template, template)

        # Test getting nonexistent template
        with self.assertRaises(KeyError):
            self.engine.get_template('Nonexistent Template')

    def test_remove_template(self):
        """Test removing a template."""
        template = ReportTemplate(
            name='Test Template',
            description='A test template'
        )
        self.engine.register_template(template)
        self.assertEqual(len(self.engine.templates), 1)

        self.engine.remove_template('Test Template')
        self.assertEqual(len(self.engine.templates), 0)

        # Test removing nonexistent template
        with self.assertRaises(KeyError):
            self.engine.remove_template('Nonexistent Template')

    @patch('analytics.reporting.EnhancedReport')
    def test_create_report(self, mock_enhanced_report):
        """Test creating a report."""
        # Mock EnhancedReport
        mock_report_instance = MagicMock()
        mock_enhanced_report.return_value = mock_report_instance

        # Register template
        template = ReportTemplate(
            name='Test Template',
            description='A test template'
        )
        self.engine.register_template(template)

        # Create report
        report = self.engine.create_report(
            title='Test Report',
            portfolio_id='TEST123',
            template_name='Test Template',
            start_date='2020-01-01',
            end_date='2020-04-10'
        )

        # Check that EnhancedReport was created with correct parameters
        mock_enhanced_report.assert_called_once_with(
            title='Test Report',
            portfolio_id='TEST123',
            template=template,
            start_date='2020-01-01',
            end_date='2020-04-10'
        )
        self.assertEqual(report, mock_report_instance)

    def test_load_default_templates(self):
        """Test loading default templates."""
        self.engine.load_default_templates()
        self.assertGreater(len(self.engine.templates), 0)
        self.assertTrue('Performance Analysis' in self.engine.templates)
        self.assertTrue('Allocation Analysis' in self.engine.templates)
        self.assertTrue('Risk Analysis' in self.engine.templates)
        self.assertTrue('Comprehensive Analysis' in self.engine.templates)


if __name__ == '__main__':
    unittest.main()