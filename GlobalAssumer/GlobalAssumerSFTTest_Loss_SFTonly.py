#!/usr/bin/env python
"""
SFT Model Loss Testing Script

This script evaluates SFT (fine-tuned) models by calculating loss metrics
on test datasets. It focuses on testing fine-tuned model performance only.

IMPORTANT: The loss calculation matches the training process by only computing
loss on the response part (after "generated rules:\n" delimiter), not the entire
input text. This ensures accurate evaluation that reflects the actual training loss.

Features:
- Load and evaluate fine-tuned SFT models
- Calculate perplexity and cross-entropy loss on response tokens only
- Support for both Spider and Bird datasets
- Generate detailed loss analysis reports
- Visualize loss metrics across multiple models
- Export results to JSON and CSV formats

Usage:
  # Test a single fine-tuned model
  python GlobalAssumerSFTTest_Loss_SFTonly.py \
    --fine_tuned_models /path/to/fine_tuned/model \
    --test_data /path/to/test_data.json \
    --output_dir /path/to/results

  # Compare multiple fine-tuned models
  python GlobalAssumerSFTTest_Loss_SFTonly.py \
    --fine_tuned_models /path/to/model1 /path/to/model2 /path/to/model3 \
    --test_data /path/to/test_data.json \
    --output_dir /path/to/results

Author: ChatTB Team
Date: 2024
"""

from __future__ import annotations

import argparse
import json
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from Process_model.SchemaInformation import SchemaInformation


@dataclass
class LossMetrics:
    """Container for loss evaluation metrics."""
    model_name: str
    total_samples: int
    total_tokens: int  # Number of response tokens (after delimiter)
    avg_loss: float
    perplexity: float
    cross_entropy_loss: float
    min_loss: float
    max_loss: float
    std_loss: float
    median_loss: float
    evaluation_time: float
    memory_usage_mb: float


class ModelLossEvaluator:
    """Evaluates model loss on test datasets."""
    
    def __init__(self, 
                 model_path: str,
                 trust_remote_code: bool = True,
                 use_quantization: bool = False,
                 device: str = "auto",
                 delimiter: str = "generated rules:\n"):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the model directory
            trust_remote_code: Whether to trust remote code
            use_quantization: Whether to use 4-bit quantization for memory efficiency
            device: Device to load model on ("auto", "cuda", "cpu")
            delimiter: Delimiter to mask labels before (default: "generated rules:\n")
        """
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.use_quantization = use_quantization
        self.device = device
        self.delimiter = delimiter
        
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization if requested
            bnb_config = None
            if self.use_quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map=self.device if self.device == "auto" else None,
                torch_dtype=torch.bfloat16 if self.use_quantization else torch.float16,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=True
            )
            
            if self.device != "auto":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def calculate_loss(self, text: str) -> Tuple[float, int]:
        """
        Calculate cross-entropy loss for the response part only (after delimiter).
        This matches the training loss calculation using CollatorMaskAfterDelimiter.
        
        Args:
            text: Input text to evaluate
            
        Returns:
            Tuple of (loss, num_tokens)
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096,
                padding=True
            )
            
            # Move to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Create labels and mask everything before the delimiter
            labels = inputs["input_ids"].clone()
            
            # Compute delimiter token ids
            delimiter_ids = self.tokenizer(self.delimiter, add_special_tokens=False)["input_ids"]
            
            if len(delimiter_ids) > 0:
                # Find delimiter in the sequence and mask everything before it
                seq = inputs["input_ids"][0].tolist()
                start_index = -1
                
                for j in range(0, len(seq) - len(delimiter_ids) + 1):
                    if seq[j:j + len(delimiter_ids)] == delimiter_ids:
                        start_index = j + len(delimiter_ids)
                        break
                
                if start_index != -1:
                    # Mask labels before the delimiter (set to -100)
                    labels[0, :start_index] = -100
                else:
                    # If delimiter not found, mask everything (no valid response)
                    labels[0, :] = -100
            
            # Calculate loss with masked labels
            with torch.no_grad():
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss.item()
            
            # Count only the unmasked tokens (response tokens)
            num_response_tokens = (labels != -100).sum().item()
            
            return loss, num_response_tokens
            
        except Exception as e:
            self.logger.warning(f"Error calculating loss for text: {e}")
            return float('inf'), 0
    
    def evaluate_dataset(self, test_data: List[Dict[str, Any]]) -> LossMetrics:
        """
        Evaluate loss on a dataset.
        
        Args:
            test_data: List of test samples with 'text' field
            
        Returns:
            LossMetrics object with evaluation results
        """
        start_time = time.time()
        
        losses = []
        total_tokens = 0
        processed_samples = 0
        
        self.logger.info(f"Evaluating {len(test_data)} samples...")
        
        for sample in tqdm(test_data, desc="Evaluating samples"):
            text = sample.get('text', '')
            if not text:
                continue
            
            loss, num_tokens = self.calculate_loss(text)
            
            if loss != float('inf'):
                losses.append(loss)
                total_tokens += num_tokens
                processed_samples += 1
        
        evaluation_time = time.time() - start_time
        
        if not losses:
            self.logger.warning("No valid loss calculations found")
            return LossMetrics(
                model_name=os.path.basename(self.model_path),
                total_samples=len(test_data),
                total_tokens=0,
                avg_loss=float('inf'),
                perplexity=float('inf'),
                cross_entropy_loss=float('inf'),
                min_loss=float('inf'),
                max_loss=float('inf'),
                std_loss=0.0,
                median_loss=float('inf'),
                evaluation_time=evaluation_time,
                memory_usage_mb=0.0
            )
        
        # Calculate metrics
        losses = np.array(losses)
        avg_loss = np.mean(losses)
        perplexity = np.exp(avg_loss)
        
        # Get memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_usage = 0.0
        
        return LossMetrics(
            model_name=os.path.basename(self.model_path),
            total_samples=len(test_data),
            total_tokens=total_tokens,
            avg_loss=avg_loss,
            perplexity=perplexity,
            cross_entropy_loss=avg_loss,
            min_loss=np.min(losses),
            max_loss=np.max(losses),
            std_loss=np.std(losses),
            median_loss=np.median(losses),
            evaluation_time=evaluation_time,
            memory_usage_mb=memory_usage
        )


class LossTestSuite:
    """Comprehensive loss testing suite for fine-tuned models."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the test suite.
        
        Args:
            output_dir: Directory to save results and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.schema_helper = SchemaInformation()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "loss_test.log"),
                logging.StreamHandler()
            ]
        )
    
    def load_test_data(self, 
                      rules_file: str,
                      db_root_path: str = "./Spider_dev/database",
                      schema_rows: int = 0,
                      max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load and prepare test data from rules file.
        
        Args:
            rules_file: Path to rules JSON file
            db_root_path: Root directory containing database files
            schema_rows: Number of sample rows to include per table
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            List of test samples with formatted prompts
        """
        self.logger.info(f"Loading test data from {rules_file}")
        
        test_data = []
        
        with open(rules_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = list(data.values())
        else:
            raise ValueError("Unsupported rules file structure")
        
        # Limit samples if requested
        if max_samples:
            items = items[:max_samples]
        
        instruction = '''
        You are an expert in analyzing database schemas and user questions to infer possible rules.  
        Rules describe **special mappings or operations** that must be followed when interpreting the question and generating SQL.  

        The output format must always be:

        rules:
        [condition]: [operation], 
        [condition]: [operation],
        ...

        Rules should be concise, accurate, and schema-faithful. You have to make sure all the table and column names belongs to the schema.
        ### Examples:

        rules:
        When answering about "heads of the departments": use table "head" instead of "departments" for counting heads.

        When the question asks for customer information: use table "Customers" instead of "customers" with exact case and quotes. If the question involves multiple tables, join "Customers_cards" as T1 with "Customers" as T2 on T1.customer_id = T2.customer_id using an inner match. If the question refers to a table named "customer" instead of "customers", use the correct table name with exact case and quotes. 

        When the question asks for students who have taken courses: join table "student" with "enrollment" on student.id = enrollment.student_id, and join with "course" on course.id = enrollment.course_id.

        ### Now process the following:
        '''
        
        for item in tqdm(items, desc="Processing test samples"):
            question = item.get("question", "").strip()
            db_id = item.get("db_id", "").strip()
            rules = item.get("rules", []) or []
            
            # Extract rules
            rule_list = [rule.strip() for rule in rules if rule.strip()]
            
            # Generate schema information if database exists
            schema = ""
            if db_id:
                db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
                if os.path.exists(db_path):
                    try:
                        schema = self.schema_helper.generate_schema_info(
                            db_path, 
                            num_rows=(schema_rows if schema_rows > 0 else None)
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to generate schema for {db_id}: {e}")
                        schema = ""
            
            # Format the target rules
            if rule_list:
                target = "\n\n".join([f"{rule}" for rule in rule_list])
            else:
                target = "No rules found."
            
            # Build the complete prompt
            text = (
                f"Instruction:{instruction}\n\n\n"
                f"Database Schema:\n{schema}\n\n"
                f"Question:\n{question}\n\n"
                f"generated rules:\n{target}"
            )
            
            test_data.append({
                "text": text,
                "question": question,
                "db_id": db_id,
                "rules": rule_list
            })
        
        self.logger.info(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def evaluate_model(self, 
                      model_path: str,
                      test_data: List[Dict[str, Any]],
                      use_quantization: bool = False,
                      delimiter: str = "generated rules:\n") -> LossMetrics:
        """
        Evaluate a single model on test data.
        
        Args:
            model_path: Path to the model
            test_data: Test dataset
            use_quantization: Whether to use quantization
            delimiter: Delimiter to mask labels before
            
        Returns:
            LossMetrics object
        """
        self.logger.info(f"Evaluating model: {model_path}")
        
        evaluator = ModelLossEvaluator(
            model_path=model_path,
            trust_remote_code=True,
            use_quantization=use_quantization,
            delimiter=delimiter,
        )
        
        metrics = evaluator.evaluate_dataset(test_data)
        
        # Clean up GPU memory
        del evaluator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return metrics
    
    def evaluate_multiple_models(self, 
                                 fine_tuned_models: List[str],
                                 test_data: List[Dict[str, Any]],
                                 use_quantization: bool = False,
                                 delimiter: str = "generated rules:\n") -> Dict[str, LossMetrics]:
        """
        Evaluate multiple fine-tuned models.
        
        Args:
            fine_tuned_models: List of paths to fine-tuned models
            test_data: Test dataset
            use_quantization: Whether to use quantization
            delimiter: Delimiter to mask labels before
            
        Returns:
            Dictionary mapping model names to LossMetrics
        """
        self.logger.info(f"Starting evaluation of {len(fine_tuned_models)} SFT model(s)...")
        
        results = {}
        
        # Evaluate each fine-tuned model
        for model_path in fine_tuned_models:
            model_name = os.path.basename(model_path)
            self.logger.info(f"Evaluating SFT model: {model_name}")
            results[model_name] = self.evaluate_model(
                model_path, test_data, use_quantization, delimiter
            )
        
        return results
    
    def generate_report(self, results: Dict[str, LossMetrics]) -> None:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Dictionary of evaluation results
        """
        self.logger.info("Generating evaluation report...")
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in results.items():
            summary_data.append({
                "Model": model_name,
                "Total Samples": metrics.total_samples,
                "Total Tokens": metrics.total_tokens,
                "Average Loss": metrics.avg_loss,
                "Perplexity": metrics.perplexity,
                "Min Loss": metrics.min_loss,
                "Max Loss": metrics.max_loss,
                "Std Loss": metrics.std_loss,
                "Median Loss": metrics.median_loss,
                "Evaluation Time (s)": metrics.evaluation_time,
                "Memory Usage (MB)": metrics.memory_usage_mb
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = self.output_dir / "loss_evaluation_summary.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Summary saved to {csv_path}")
        
        # Save detailed results to JSON
        json_results = {}
        for model_name, metrics in results.items():
            json_results[model_name] = {
                "model_name": metrics.model_name,
                "total_samples": metrics.total_samples,
                "total_tokens": metrics.total_tokens,
                "avg_loss": metrics.avg_loss,
                "perplexity": metrics.perplexity,
                "cross_entropy_loss": metrics.cross_entropy_loss,
                "min_loss": metrics.min_loss,
                "max_loss": metrics.max_loss,
                "std_loss": metrics.std_loss,
                "median_loss": metrics.median_loss,
                "evaluation_time": metrics.evaluation_time,
                "memory_usage_mb": metrics.memory_usage_mb
            }
        
        json_path = self.output_dir / "detailed_loss_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Detailed results saved to {json_path}")
        
        # Generate visualizations
        self._create_visualizations(df, results)
        
        # Print summary
        self._print_summary(df, results)
    
    def _create_visualizations(self, df: pd.DataFrame, results: Dict[str, LossMetrics]) -> None:
        """Create visualization plots for SFT models."""
        plt.style.use('default')
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SFT Model Loss Analysis', fontsize=16, fontweight='bold')
        
        # 1. Average Loss Comparison
        ax1 = axes[0, 0]
        models = df['Model']
        avg_losses = df['Average Loss']
        bars = ax1.bar(models, avg_losses, color='blue', alpha=0.7)
        ax1.set_title('Average Loss Comparison')
        ax1.set_ylabel('Average Loss')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_losses):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Perplexity Comparison
        ax2 = axes[0, 1]
        perplexities = df['Perplexity']
        bars = ax2.bar(models, perplexities, color='green', alpha=0.7)
        ax2.set_title('Perplexity Comparison')
        ax2.set_ylabel('Perplexity')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, perplexities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Loss Distribution (Min, Max, Median)
        ax3 = axes[1, 0]
        x = np.arange(len(models))
        width = 0.25
        
        ax3.bar(x - width, df['Min Loss'], width, label='Min Loss', alpha=0.8, color='lightblue')
        ax3.bar(x, df['Median Loss'], width, label='Median Loss', alpha=0.8, color='blue')
        ax3.bar(x + width, df['Max Loss'], width, label='Max Loss', alpha=0.8, color='darkblue')
        
        ax3.set_title('Loss Distribution (Min/Median/Max)')
        ax3.set_ylabel('Loss Value')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        
        # 4. Evaluation Time and Memory Usage
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        
        # Evaluation time
        time_bars = ax4.bar([x - 0.2 for x in range(len(models))], df['Evaluation Time (s)'], 
                          0.4, label='Evaluation Time (s)', alpha=0.8, color='orange')
        ax4.set_ylabel('Evaluation Time (seconds)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Memory usage
        memory_bars = ax4_twin.bar([x + 0.2 for x in range(len(models))], df['Memory Usage (MB)'], 
                                 0.4, label='Memory Usage (MB)', alpha=0.8, color='purple')
        ax4_twin.set_ylabel('Memory Usage (MB)')
        
        ax4.set_title('Performance Metrics')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels(models, rotation=45)
        
        # Add legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "loss_analysis_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {plot_path}")
    
    def _print_summary(self, df: pd.DataFrame, results: Dict[str, LossMetrics]) -> None:
        """Print evaluation summary to console."""
        print("\n" + "="*80)
        print("SFT MODEL LOSS EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total test samples: {df['Total Samples'].iloc[0]}")
        print(f"   Total tokens evaluated: {df['Total Tokens'].sum():,}")
        print(f"   Number of SFT models evaluated: {len(df)}")
        
        print(f"\n🏆 Model Performance Ranking (by Average Loss):")
        sorted_df = df.sort_values('Average Loss')
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            print(f"   {i}. {row['Model']}: {row['Average Loss']:.4f} (Perplexity: {row['Perplexity']:.2f})")
        
        print(f"\n📊 Best Model Statistics:")
        best_model_idx = df['Average Loss'].idxmin()
        best_model = df.loc[best_model_idx]
        print(f"   Model: {best_model['Model']}")
        print(f"   Average Loss: {best_model['Average Loss']:.4f}")
        print(f"   Perplexity: {best_model['Perplexity']:.2f}")
        print(f"   Min Loss: {best_model['Min Loss']:.4f}")
        print(f"   Max Loss: {best_model['Max Loss']:.4f}")
        print(f"   Std Loss: {best_model['Std Loss']:.4f}")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"   Fastest evaluation: {df['Evaluation Time (s)'].min():.1f}s ({df.loc[df['Evaluation Time (s)'].idxmin(), 'Model']})")
        print(f"   Lowest memory usage: {df['Memory Usage (MB)'].min():.1f}MB ({df.loc[df['Memory Usage (MB)'].idxmin(), 'Model']})")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print("="*80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SFT (fine-tuned) models using loss metrics"
    )
    
    # Model configuration
    parser.add_argument("--fine_tuned_models", type=str, nargs="+", required=True,
                       help="Path(s) to fine-tuned SFT model(s)")
    
    # Data configuration
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data JSON file (condensed_rules.json)")
    parser.add_argument("--db_root_path", type=str, default="./Spider_dev/database",
                       help="Root directory containing database files")
    parser.add_argument("--schema_rows", type=int, default=0,
                       help="Number of sample rows to include per table")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of test samples to evaluate")
    
    # Evaluation configuration
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save evaluation results")
    parser.add_argument("--use_quantization", action="store_true",
                       help="Use 4-bit quantization for memory efficiency")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to run evaluation on")
    parser.add_argument("--delimiter", type=str, default="generated rules:\n",
                       help="Delimiter to mask labels before (default: 'generated rules:\\n')")
    
    return parser.parse_args()


def main():
    """Main function to run SFT model loss evaluation."""
    args = parse_args()
    
    # Initialize test suite
    test_suite = LossTestSuite(args.output_dir)
    
    # Load test data
    test_data = test_suite.load_test_data(
        rules_file=args.test_data,
        db_root_path=args.db_root_path,
        schema_rows=args.schema_rows,
        max_samples=args.max_samples
    )
    
    if not test_data:
        raise ValueError("No test data loaded. Check your test data file and paths.")
    
    # Evaluate SFT models
    results = test_suite.evaluate_multiple_models(
        fine_tuned_models=args.fine_tuned_models,
        test_data=test_data,
        use_quantization=args.use_quantization,
        delimiter=args.delimiter
    )
    
    # Generate comprehensive report
    test_suite.generate_report(results)
    
    print(f"\n✅ SFT model loss evaluation completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
