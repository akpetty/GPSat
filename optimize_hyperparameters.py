import os
import sys
import json
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
from datetime import datetime
import itertools
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparameter_optimization.log'),
        logging.StreamHandler()
    ]
)

class HyperparameterOptimizer:
    def __init__(self, base_config):
        """
        Initialize the hyperparameter optimizer
        
        Parameters:
        -----------
        base_config : dict
            Base configuration dictionary with default values
        """
        self.base_config = base_config
        self.results = []
        self.start_time = None
        self.total_combinations = None
        
    def create_param_grid(self):
        """Define the parameter grid for optimization"""
        return {
            'length_scale': [100_000, 200_000, 300_000],  # Length scale for GP
            'noise_std': [0.01, 0.05, 0.1, 0.2],  # Extended noise range
            'expert_spacing': [300_000, 500_000, 700_000],  # Expert spacing
            'training_radius': [300_000, 500_000, 700_000],  # Training radius
            'sic_cutoff': [0.1, 0.15, 0.2],  # SIC cutoff
            'sic_coarsen_factor': [2, 4, 6]  # SIC coarsening factor
        }
    
    def get_system_info(self):
        """Get current system resource usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'memory_used_gb': memory_info.rss / (1024 * 1024 * 1024)
        }
    
    def run_training(self, params):
        """
        Run the training script with given parameters
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters to use for this run
            
        Returns:
        --------
        dict
            Dictionary containing score and timing information
        """
        # Create a temporary config file
        config = self.base_config.copy()
        config.update(params)
        
        # Create a unique output directory for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config['output_dir'] = f'./results/optimization_{timestamp}'
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Save config to file
        config_file = os.path.join(config['output_dir'], 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Get system info before run
        pre_run_info = self.get_system_info()
        
        # Run the training script
        try:
            cmd = [
                'python', 'IS2_GPSat_train.py',
                '--config', config_file,
                '--num_days', str(config['num_days_before_after']),
                '--test_name', f'optimization_{timestamp}'
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            end_time = time.time()
            
            # Get system info after run
            post_run_info = self.get_system_info()
            
            if result.returncode != 0:
                logging.error(f"Error running training: {result.stderr}")
                return {
                    'score': float('inf'),
                    'duration': end_time - start_time,
                    'error': result.stderr,
                    'system_info': {
                        'pre_run': pre_run_info,
                        'post_run': post_run_info
                    }
                }
            
            # Parse the output to get the MSE
            try:
                mse = float(result.stdout.split('MSE: ')[-1].split('\n')[0])
                return {
                    'score': mse,
                    'duration': end_time - start_time,
                    'system_info': {
                        'pre_run': pre_run_info,
                        'post_run': post_run_info
                    }
                }
            except:
                logging.error("Could not parse MSE from output")
                return {
                    'score': float('inf'),
                    'duration': end_time - start_time,
                    'error': "Could not parse MSE from output",
                    'system_info': {
                        'pre_run': pre_run_info,
                        'post_run': post_run_info
                    }
                }
                
        except Exception as e:
            logging.error(f"Exception running training: {str(e)}")
            return {
                'score': float('inf'),
                'duration': 0,
                'error': str(e),
                'system_info': {
                    'pre_run': pre_run_info,
                    'post_run': pre_run_info
                }
            }
    
    def optimize(self):
        """Run the hyperparameter optimization"""
        param_grid = self.create_param_grid()
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        self.total_combinations = len(param_combinations)
        self.start_time = time.time()
        
        logging.info(f"Starting optimization with {self.total_combinations} combinations")
        
        best_score = float('inf')
        best_params = None
        
        # Create progress bar
        pbar = tqdm(total=self.total_combinations, desc="Optimizing parameters")
        
        for i, params in enumerate(param_combinations):
            logging.info(f"Testing combination {i+1}/{self.total_combinations}")
            logging.info(f"Parameters: {params}")
            
            result = self.run_training(params)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'best_score': f"{best_score:.4f}",
                'current_score': f"{result['score']:.4f}",
                'duration': f"{result['duration']:.1f}s"
            })
            
            # Calculate estimated time remaining
            elapsed_time = time.time() - self.start_time
            avg_time_per_run = elapsed_time / (i + 1)
            remaining_runs = self.total_combinations - (i + 1)
            estimated_time_remaining = avg_time_per_run * remaining_runs
            
            # Add timing information to result
            result['params'] = params
            result['combination_number'] = i + 1
            result['total_combinations'] = self.total_combinations
            result['elapsed_time'] = elapsed_time
            result['estimated_time_remaining'] = estimated_time_remaining
            
            self.results.append(result)
            
            if result['score'] < best_score:
                best_score = result['score']
                best_params = params
                logging.info(f"New best parameters found! Score: {best_score:.4f}")
        
        pbar.close()
        
        # Save detailed results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('hyperparameter_optimization_results.csv', index=False)
        
        # Save best parameters
        with open('best_parameters.json', 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_score': best_score,
                'total_time': time.time() - self.start_time,
                'total_combinations': self.total_combinations
            }, f, indent=4)
        
        # Generate summary statistics
        summary = {
            'best_params': best_params,
            'best_score': best_score,
            'total_time': time.time() - self.start_time,
            'total_combinations': self.total_combinations,
            'avg_time_per_run': np.mean([r['duration'] for r in self.results]),
            'min_time_per_run': np.min([r['duration'] for r in self.results]),
            'max_time_per_run': np.max([r['duration'] for r in self.results]),
            'successful_runs': sum(1 for r in self.results if r['score'] != float('inf')),
            'failed_runs': sum(1 for r in self.results if r['score'] == float('inf'))
        }
        
        # Save summary
        with open('optimization_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        logging.info("Optimization complete!")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best score: {best_score:.4f}")
        logging.info(f"Total time: {summary['total_time']:.2f} seconds")
        logging.info(f"Average time per run: {summary['avg_time_per_run']:.2f} seconds")
        
        return best_params, best_score, summary

def main():
    # Base configuration
    base_config = {
        'target_date': '2019-01-15',
        'num_days_before_after': 1,
        'beam': 'bnum1',
        'sic_cutoff': 0.15,
        'sic_coarsen_factor': 4,
        'noise_std': 0.1,
        'expert_spacing': 500_000,
        'training_radius': 500_000,
        'length_scale': 200_000
    }
    
    # Create optimizer and run optimization
    optimizer = HyperparameterOptimizer(base_config)
    best_params, best_score, summary = optimizer.optimize()
    
    print("\nOptimization Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    print(f"\nTiming Information:")
    print(f"Total time: {summary['total_time']:.2f} seconds")
    print(f"Average time per run: {summary['avg_time_per_run']:.2f} seconds")
    print(f"Min time per run: {summary['min_time_per_run']:.2f} seconds")
    print(f"Max time per run: {summary['max_time_per_run']:.2f} seconds")
    print(f"\nRun Statistics:")
    print(f"Total combinations: {summary['total_combinations']}")
    print(f"Successful runs: {summary['successful_runs']}")
    print(f"Failed runs: {summary['failed_runs']}")
    print("\nDetailed results saved in:")
    print("- hyperparameter_optimization_results.csv")
    print("- best_parameters.json")
    print("- optimization_summary.json")
    print("- hyperparameter_optimization.log")

if __name__ == "__main__":
    main() 