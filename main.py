# main.py - FastAPI service wrapper for Crystal Ball MVP
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union, Callable
import json
import uuid
import os
import tempfile
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Cloud Run
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Import all the required libraries
import numpy as np
import seaborn as sns
from scipy import stats
from abc import ABC, abstractmethod
import warnings
from dataclasses import dataclass
from enum import Enum
import threading

# Configure plotting for cloud environment
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.ioff()  # Turn off interactive mode

class DistributionType(Enum):
    """Enumeration of supported probability distributions"""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    BETA = "beta"
    GAMMA = "gamma"
    POISSON = "poisson"
    BINOMIAL = "binomial"
    DISCRETE = "discrete"

@dataclass
class SimulationResults:
    """Container for simulation results"""
    values: np.ndarray
    statistics: Dict[str, float]
    percentiles: Dict[str, float]
    confidence_intervals: Dict[str, tuple]

class ProbabilityDistribution(ABC):
    """Abstract base class for probability distributions"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def sample(self, size: int) -> np.ndarray:
        """Generate random samples from the distribution"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return distribution parameters"""
        pass

class NormalDistribution(ProbabilityDistribution):
    """Normal distribution implementation"""
    
    def __init__(self, mean: float, std: float):
        super().__init__("Normal")
        self.mean = mean
        self.std = std
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.normal(self.mean, self.std, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

class UniformDistribution(ProbabilityDistribution):
    """Uniform distribution implementation"""
    
    def __init__(self, min_val: float, max_val: float):
        super().__init__("Uniform")
        self.min_val = min_val
        self.max_val = max_val
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.uniform(self.min_val, self.max_val, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"min": self.min_val, "max": self.max_val}

class TriangularDistribution(ProbabilityDistribution):
    """Triangular distribution implementation"""
    
    def __init__(self, min_val: float, mode: float, max_val: float):
        super().__init__("Triangular")
        self.min_val = min_val
        self.mode = mode
        self.max_val = max_val
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.triangular(self.min_val, self.mode, self.max_val, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"min": self.min_val, "mode": self.mode, "max": self.max_val}

class LogNormalDistribution(ProbabilityDistribution):
    """Log-normal distribution implementation"""
    
    def __init__(self, mu: float, sigma: float):
        super().__init__("Log-Normal")
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.lognormal(self.mu, self.sigma, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"mu": self.mu, "sigma": self.sigma}

class BetaDistribution(ProbabilityDistribution):
    """Beta distribution implementation"""
    
    def __init__(self, alpha: float, beta: float):
        super().__init__("Beta")
        self.alpha = alpha
        self.beta = beta
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.beta(self.alpha, self.beta, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta}

class GammaDistribution(ProbabilityDistribution):
    """Gamma distribution implementation"""
    
    def __init__(self, shape: float, scale: float):
        super().__init__("Gamma")
        self.shape = shape
        self.scale = scale
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.gamma(self.shape, self.scale, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"shape": self.shape, "scale": self.scale}

class ExponentialDistribution(ProbabilityDistribution):
    """Exponential distribution implementation"""
    
    def __init__(self, scale: float):
        super().__init__("Exponential")
        self.scale = scale
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.exponential(self.scale, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"scale": self.scale}

class PoissonDistribution(ProbabilityDistribution):
    """Poisson distribution implementation"""
    
    def __init__(self, lam: float):
        super().__init__("Poisson")
        self.lam = lam
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.poisson(self.lam, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"lambda": self.lam}

class BinomialDistribution(ProbabilityDistribution):
    """Binomial distribution implementation"""
    
    def __init__(self, n: int, p: float):
        super().__init__("Binomial")
        self.n = n
        self.p = p
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.binomial(self.n, self.p, size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"n": self.n, "p": self.p}

class DiscreteDistribution(ProbabilityDistribution):
    """Discrete/Empirical distribution implementation"""
    
    def __init__(self, values: List[float], probabilities: List[float]):
        super().__init__("Discrete")
        self.values = np.array(values)
        self.probabilities = np.array(probabilities)
        # Normalize probabilities
        self.probabilities = self.probabilities / np.sum(self.probabilities)
    
    def sample(self, size: int) -> np.ndarray:
        return np.random.choice(self.values, size=size, p=self.probabilities)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"values": self.values.tolist(), "probabilities": self.probabilities.tolist()}

class Assumption:
    """Represents an uncertain input variable with a probability distribution"""
    
    def __init__(self, name: str, distribution: ProbabilityDistribution, description: str = ""):
        self.name = name
        self.distribution = distribution
        self.description = description
        self.samples = None
    
    def generate_samples(self, size: int) -> np.ndarray:
        """Generate random samples for this assumption"""
        self.samples = self.distribution.sample(size)
        return self.samples
    
    def __repr__(self):
        return f"Assumption(name='{self.name}', distribution={self.distribution.name})"

class Forecast:
    """Represents an output variable to be analyzed"""
    
    def __init__(self, name: str, formula: Callable, description: str = ""):
        self.name = name
        self.formula = formula
        self.description = description
        self.results = None
    
    def calculate(self, assumptions_samples: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate forecast values using the formula and assumption samples"""
        try:
            self.results = self.formula(assumptions_samples)
            return self.results
        except Exception as e:
            raise ValueError(f"Error calculating forecast '{self.name}': {str(e)}")
    
    def __repr__(self):
        return f"Forecast(name='{self.name}')"

class SensitivityAnalyzer:
    """Performs sensitivity analysis on model inputs"""
    
    @staticmethod
    def rank_correlations(assumptions: Dict[str, Assumption], 
                         forecast: Forecast) -> pd.DataFrame:
        """Calculate rank correlations between assumptions and forecast"""
        if forecast.results is None:
            raise ValueError("Forecast must be calculated before sensitivity analysis")
        
        correlations = []
        forecast_ranks = stats.rankdata(forecast.results)
        
        for name, assumption in assumptions.items():
            if assumption.samples is not None:
                assumption_ranks = stats.rankdata(assumption.samples)
                correlation = np.corrcoef(assumption_ranks, forecast_ranks)[0, 1]
                correlations.append({
                    'Assumption': name,
                    'Correlation': correlation,
                    'Absolute_Correlation': abs(correlation)
                })
        
        df = pd.DataFrame(correlations)
        return df.sort_values('Absolute_Correlation', ascending=False)
    
    @staticmethod
    def contribution_to_variance(assumptions: Dict[str, Assumption], 
                               forecast: Forecast) -> pd.DataFrame:
        """Calculate contribution of each assumption to forecast variance"""
        if forecast.results is None:
            raise ValueError("Forecast must be calculated before sensitivity analysis")
        
        contributions = []
        forecast_var = np.var(forecast.results)
        
        for name, assumption in assumptions.items():
            if assumption.samples is not None:
                covariance = np.cov(assumption.samples, forecast.results)[0, 1]
                assumption_var = np.var(assumption.samples)
                if assumption_var > 0:
                    contribution = (covariance ** 2) / (assumption_var * forecast_var)
                    contributions.append({
                        'Assumption': name,
                        'Contribution': contribution,
                        'Percentage': contribution * 100
                    })
        
        df = pd.DataFrame(contributions)
        return df.sort_values('Contribution', ascending=False)

class TornadoEngine:
    """OCB-style tornado and spider analysis"""
    
    @staticmethod
    def analyze(cb: 'CrystalBallMVP', forecast_name: str,
                low_pct: float = 10, high_pct: float = 90, test_points: int = 5,
                baseline_pct: float = 50) -> pd.DataFrame:
        """
        Perform tornado analysis by varying each assumption across percentiles
        while holding others fixed at baseline
        """
        if forecast_name not in cb.forecasts:
            raise ValueError(f"Forecast '{forecast_name}' not found")
        
        forecast = cb.forecasts[forecast_name]
        
        # Calculate baseline values for all assumptions
        baseline_values = {}
        for name, assumption in cb.assumptions.items():
            if assumption.samples is not None:
                baseline_values[name] = np.percentile(assumption.samples, baseline_pct)
            else:
                raise ValueError(f"Assumption '{name}' has no samples. Run simulation first.")
        
        # Generate test percentiles
        test_percentiles = np.linspace(low_pct, high_pct, test_points)
        
        results = []
        
        for assumption_name, assumption in cb.assumptions.items():
            assumption_results = []
            
            for pct in test_percentiles:
                # Create environment with all assumptions at baseline except the one being varied
                test_env = {}
                for name, baseline_val in baseline_values.items():
                    if name == assumption_name:
                        # Vary this assumption
                        test_value = np.percentile(assumption.samples, pct)
                        test_env[name] = np.full(cb.trials, test_value)
                    else:
                        # Keep others at baseline
                        test_env[name] = np.full(cb.trials, baseline_val)
                
                # Calculate forecast for this scenario
                forecast_result = forecast.formula(test_env)
                forecast_mean = float(np.mean(forecast_result))
                
                assumption_results.append({
                    'Assumption': assumption_name,
                    'Percentile': pct,
                    'Forecast': forecast_name,
                    'Value': forecast_mean
                })
            
            results.extend(assumption_results)
        
        df = pd.DataFrame(results)
        
        # Calculate swing (range) for each assumption
        swing_data = []
        for assumption_name in cb.assumptions.keys():
            assumption_data = df[df['Assumption'] == assumption_name]
            swing = assumption_data['Value'].max() - assumption_data['Value'].min()
            swing_data.append({
                'Assumption': assumption_name,
                'Swing': swing,
                'Low_Value': assumption_data['Value'].min(),
                'High_Value': assumption_data['Value'].max()
            })
        
        swing_df = pd.DataFrame(swing_data)
        return df.merge(swing_df, on='Assumption', how='left')

class CrystalBallMVP:
    """Main Monte Carlo simulation framework - Crystal Ball MVP"""
    
    def __init__(self):
        self.assumptions = {}
        self.forecasts = {}
        self.trials = 10000
        self.random_seed = None
        self.results = {}
        
    def set_trials(self, trials: int):
        """Set the number of Monte Carlo trials"""
        if trials < 100:
            warnings.warn("Number of trials is very low. Consider using at least 1000 trials.")
        self.trials = trials
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible results"""
        self.random_seed = seed
        np.random.seed(seed)
    
    def add_assumption(self, assumption: Assumption):
        """Add an assumption (uncertain input) to the model"""
        self.assumptions[assumption.name] = assumption
    
    def add_forecast(self, forecast: Forecast):
        """Add a forecast (output) to the model"""
        self.forecasts[forecast.name] = forecast
    
    def run_simulation(self):
        """Execute the Monte Carlo simulation"""
        print(f"Running Monte Carlo simulation with {self.trials} trials...")
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Generate samples for all assumptions
        assumption_samples = {}
        for name, assumption in self.assumptions.items():
            assumption_samples[name] = assumption.generate_samples(self.trials)
        
        # Calculate forecasts
        self.results = {}
        for name, forecast in self.forecasts.items():
            print(f"Calculating forecast: {name}")
            forecast_values = forecast.calculate(assumption_samples)
            self.results[name] = self._create_simulation_results(forecast_values)
        
        print("Simulation completed successfully!")
    
    def trials_dataframe(self) -> pd.DataFrame:
        """Return trial-level data in wide format for Google Sheets"""
        if not self.results:
            raise ValueError("No simulation results found. Run simulation first.")
        
        data = {"Trial": np.arange(1, self.trials + 1)}
        
        # Add assumption samples
        for name, assumption in self.assumptions.items():
            if assumption.samples is not None:
                data[name] = assumption.samples
            else:
                raise ValueError(f"Assumption '{name}' has no samples")
        
        # Add forecast results
        for name, forecast in self.forecasts.items():
            if forecast.results is not None:
                data[name] = forecast.results
            else:
                raise ValueError(f"Forecast '{name}' has no results")
        
        return pd.DataFrame(data)
    
    def _create_simulation_results(self, values: np.ndarray) -> SimulationResults:
        """Create simulation results object with statistics"""
        statistics = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'variance': np.var(values),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values)
        }
        
        percentiles = {
            f'p{p}': np.percentile(values, p) 
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
        
        confidence_intervals = {
            '90%': (np.percentile(values, 5), np.percentile(values, 95)),
            '95%': (np.percentile(values, 2.5), np.percentile(values, 97.5)),
            '99%': (np.percentile(values, 0.5), np.percentile(values, 99.5))
        }
        
        return SimulationResults(values, statistics, percentiles, confidence_intervals)
    
    def get_statistics(self, forecast_name: str) -> Dict[str, float]:
        """Get descriptive statistics for a forecast"""
        if forecast_name not in self.results:
            raise ValueError(f"Forecast '{forecast_name}' not found in results")
        return self.results[forecast_name].statistics
    
    def get_percentiles(self, forecast_name: str) -> Dict[str, float]:
        """Get percentiles for a forecast"""
        if forecast_name not in self.results:
            raise ValueError(f"Forecast '{forecast_name}' not found in results")
        return self.results[forecast_name].percentiles
    
    def sensitivity_analysis(self, forecast_name: str) -> Dict[str, pd.DataFrame]:
        """Perform sensitivity analysis for a forecast"""
        if forecast_name not in self.forecasts:
            raise ValueError(f"Forecast '{forecast_name}' not found")
        
        forecast = self.forecasts[forecast_name]
        
        return {
            'correlations': SensitivityAnalyzer.rank_correlations(self.assumptions, forecast),
            'variance_contributions': SensitivityAnalyzer.contribution_to_variance(self.assumptions, forecast)
        }
    
    def tornado_analysis(self, forecast_name: str, low_pct: float = 10, high_pct: float = 90, 
                        test_points: int = 5, baseline_pct: float = 50) -> pd.DataFrame:
        """Perform OCB-style tornado analysis"""
        return TornadoEngine.analyze(self, forecast_name, low_pct, high_pct, test_points, baseline_pct)
    
    def binned_and_cdf_data(self, forecast_name: str, bins: int = 50) -> Dict[str, Any]:
        """Get binned histogram and CDF data for overlay/cumulative charts"""
        if forecast_name not in self.results:
            raise ValueError(f"Forecast '{forecast_name}' not found in results")
        
        values = self.results[forecast_name].values
        
        # Histogram data
        counts, edges = np.histogram(values, bins=bins)
        bin_midpoints = (edges[:-1] + edges[1:]) / 2
        
        # CDF data
        sorted_values = np.sort(values)
        cdf_y = np.arange(1, len(values) + 1) / len(values)
        
        return {
            "bin_midpoints": bin_midpoints.tolist(),
            "bin_counts": counts.tolist(),
            "bin_edges": edges.tolist(),
            "cdf_x": sorted_values.tolist(),
            "cdf_y": cdf_y.tolist(),
            "total_trials": len(values)
        }
    
    def plot_forecast_histogram(self, forecast_name: str, bins: int = 50, 
                              figsize: tuple = (10, 6), show_stats: bool = True):
        """Plot histogram of forecast results and return as base64 string"""
        if forecast_name not in self.results:
            raise ValueError(f"Forecast '{forecast_name}' not found in results")
        
        results = self.results[forecast_name]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.hist(results.values, bins=bins, alpha=0.7, color='skyblue', 
                edgecolor='black', density=True)
        
        # Add mean line
        mean_val = results.statistics['mean']
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.2f}')
        
        # Add percentile lines
        p5 = results.percentiles['p5']
        p95 = results.percentiles['p95']
        ax.axvline(p5, color='orange', linestyle='--', alpha=0.7, 
                   label=f'5th Percentile: {p5:.2f}')
        ax.axvline(p95, color='orange', linestyle='--', alpha=0.7, 
                   label=f'95th Percentile: {p95:.2f}')
        
        ax.set_title(f'Monte Carlo Forecast: {forecast_name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if show_stats:
            stats_text = f"""Statistics:
Mean: {results.statistics['mean']:.3f}
Std Dev: {results.statistics['std']:.3f}
Min: {results.statistics['min']:.3f}
Max: {results.statistics['max']:.3f}
Trials: {len(results.values)}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def plot_sensitivity_chart(self, forecast_name: str, figsize: tuple = (10, 6)):
        """Plot sensitivity analysis chart and return as base64 string"""
        sensitivity = self.sensitivity_analysis(forecast_name)
        correlations = sensitivity['correlations'].head(10)  # Top 10
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = range(len(correlations))
        bars = ax.barh(y_pos, correlations['Correlation'], 
                       color=['green' if x > 0 else 'red' for x in correlations['Correlation']])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(correlations['Assumption'])
        ax.set_xlabel('Rank Correlation')
        ax.set_title(f'Sensitivity Analysis: {forecast_name}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left' if width > 0 else 'right', va='center')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def generate_report(self, forecast_name: str) -> str:
        """Generate a comprehensive text report"""
        if forecast_name not in self.results:
            raise ValueError(f"Forecast '{forecast_name}' not found in results")
        
        results = self.results[forecast_name]
        sensitivity = self.sensitivity_analysis(forecast_name)
        
        report = f"""
MONTE CARLO SIMULATION REPORT
============================
Forecast: {forecast_name}
Trials: {len(results.values)}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
------------------
Mean:           {results.statistics['mean']:.6f}
Standard Dev:   {results.statistics['std']:.6f}
Variance:       {results.statistics['variance']:.6f}
Minimum:        {results.statistics['min']:.6f}
Maximum:        {results.statistics['max']:.6f}
Median:         {results.statistics['median']:.6f}
Skewness:       {results.statistics['skewness']:.6f}
Kurtosis:       {results.statistics['kurtosis']:.6f}

PERCENTILES
-----------
1st:    {results.percentiles['p1']:.6f}
5th:    {results.percentiles['p5']:.6f}
10th:   {results.percentiles['p10']:.6f}
25th:   {results.percentiles['p25']:.6f}
50th:   {results.percentiles['p50']:.6f}
75th:   {results.percentiles['p75']:.6f}
90th:   {results.percentiles['p90']:.6f}
95th:   {results.percentiles['p95']:.6f}
99th:   {results.percentiles['p99']:.6f}

CONFIDENCE INTERVALS
--------------------
90% CI: [{results.confidence_intervals['90%'][0]:.6f}, {results.confidence_intervals['90%'][1]:.6f}]
95% CI: [{results.confidence_intervals['95%'][0]:.6f}, {results.confidence_intervals['95%'][1]:.6f}]
99% CI: [{results.confidence_intervals['99%'][0]:.6f}, {results.confidence_intervals['99%'][1]:.6f}]

SENSITIVITY ANALYSIS - TOP 5 CORRELATIONS
------------------------------------------
"""
        
        top_correlations = sensitivity['correlations'].head(5)
        for _, row in top_correlations.iterrows():
            report += f"{row['Assumption']:20} {row['Correlation']:8.4f}\n"
        
        report += """
ASSUMPTIONS
-----------
"""
        for name, assumption in self.assumptions.items():
            params = assumption.distribution.get_parameters()
            report += f"{name:20} {assumption.distribution.name:15} {str(params)}\n"
        
        return report
    
    def export_results(self, filename: str, format: str = 'excel'):
        """Export results to Excel or CSV"""
        if format.lower() == 'excel':
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for forecast_name, results in self.results.items():
                    summary_data.append({
                        'Forecast': forecast_name,
                        'Mean': results.statistics['mean'],
                        'Std_Dev': results.statistics['std'],
                        'Min': results.statistics['min'],
                        'Max': results.statistics['max'],
                        'P5': results.percentiles['p5'],
                        'P95': results.percentiles['p95']
                    })
                
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Trial-level data sheet
                trials_df = self.trials_dataframe()
                trials_df.to_excel(writer, sheet_name='Trials', index=False)
                
                # Individual forecast sheets
                for forecast_name, results in self.results.items():
                    df = pd.DataFrame({
                        'Trial': range(1, len(results.values) + 1),
                        'Value': results.values
                    })
                    df.to_excel(writer, sheet_name=f'{forecast_name}', index=False)
        
        elif format.lower() == 'csv':
            # Export summary to CSV
            summary_data = []
            for forecast_name, results in self.results.items():
                summary_data.append({
                    'Forecast': forecast_name,
                    'Mean': results.statistics['mean'],
                    'Std_Dev': results.statistics['std'],
                    'Min': results.statistics['min'],
                    'Max': results.statistics['max'],
                    'P5': results.percentiles['p5'],
                    'P95': results.percentiles['p95']
                })
            
            pd.DataFrame(summary_data).to_csv(filename, index=False)

# Factory class for creating distributions
class DistributionFactory:
    """Factory class for creating probability distributions"""
    
    @staticmethod
    def create_normal(mean: float, std: float) -> NormalDistribution:
        return NormalDistribution(mean, std)
    
    @staticmethod
    def create_uniform(min_val: float, max_val: float) -> UniformDistribution:
        return UniformDistribution(min_val, max_val)
    
    @staticmethod
    def create_triangular(min_val: float, mode: float, max_val: float) -> TriangularDistribution:
        return
