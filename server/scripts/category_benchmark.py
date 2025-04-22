#!/usr/bin/env python3
import csv
import json
import random
import requests
import time
from typing import List, Dict, Set, Tuple
from datetime import datetime
import argparse


class PaperCategorizationBenchmark:
    def __init__(
        self, 
        csv_path: str, 
        api_endpoint: str,
        sample_size: int = 50,
        random_seed: int = 42,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            csv_path: Path to the CSV file containing paper data
            api_endpoint: URL for the API endpoint
            sample_size: Number of papers to sample for benchmarking
            random_seed: Random seed for reproducibility
            max_retries: Maximum number of API call retries
            retry_delay: Delay between retries in seconds
        """
        self.csv_path = csv_path
        self.api_endpoint = api_endpoint
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.results = []
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
    def load_papers(self) -> List[Dict]:
        """Load and parse papers from the CSV file."""
        papers = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Parse the categories into a set for easier comparison
                if 'categories' in row:
                    row['categories_set'] = set(row['categories'].split(' '))
                papers.append(row)
                
        print(f"Loaded {len(papers)} papers from {self.csv_path}")
        return papers
    
    def sample_papers(self, papers: List[Dict]) -> List[Dict]:
        """Sample a subset of papers for benchmarking."""
        sample_size = min(self.sample_size, len(papers))
        sampled_papers = random.sample(papers, sample_size)
        print(f"Sampled {len(sampled_papers)} papers for benchmarking")
        return sampled_papers
    
    def construct_prompt(self, paper: Dict) -> Dict:
        """Construct the API request payload for a paper."""
        # Combine title and abstract for the paper content
        paper_content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        
        return {
            "paper_content": paper_content,
            "prompt": "summarize"
        }
    
    def call_api(self, payload: Dict) -> Dict:
        """Call the API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < self.max_retries - 1:
                    print(f"Error calling API: {e}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to call API after {self.max_retries} attempts: {e}")
                    raise
    
    def evaluate_response(self, paper: Dict, api_response: Dict) -> Dict:
        """Evaluate if the API response contains the correct categories."""
        # Extract the category codes from the API response
        predicted_categories = set()
        for category in api_response.get("categories", []):
            predicted_categories.add(category.get("code", ""))
        
        # Get the ground truth categories
        true_categories = paper.get("categories_set", set())
        
        # Calculate precision and recall
        correct_predictions = predicted_categories.intersection(true_categories)
        precision = len(correct_predictions) / len(predicted_categories) if predicted_categories else 0
        recall = len(correct_predictions) / len(true_categories) if true_categories else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Check for any correct with none wrong
        any_correct_none_wrong = len(correct_predictions) > 0 and len(predicted_categories - true_categories) == 0
        
        # Build result object
        result = {
            "paper_id": paper.get("paper_id", ""),
            "title": paper.get("title", ""),
            "true_categories": list(true_categories),
            "predicted_categories": list(predicted_categories),
            "correct_categories": list(correct_predictions),
            "missing_categories": list(true_categories - predicted_categories),
            "extra_categories": list(predicted_categories - true_categories),
            "all_categories_correct": predicted_categories == true_categories,
            "any_category_correct": len(correct_predictions) > 0,
            "any_correct_none_wrong": any_correct_none_wrong,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        return result
    
    def run_benchmark(self) -> Dict:
        """Run the benchmark and return results."""
        start_time = time.time()
        
        # Load and sample papers
        all_papers = self.load_papers()
        benchmark_papers = self.sample_papers(all_papers)
        
        # Process each paper
        for i, paper in enumerate(benchmark_papers):
            print(f"Processing paper {i+1}/{len(benchmark_papers)}: {paper.get('title', '')[:50]}...")
            
            try:
                # Prepare and send API request
                payload = self.construct_prompt(paper)
                api_response = self.call_api(payload)
                
                # Evaluate the response
                result = self.evaluate_response(paper, api_response)
                self.results.append(result)
                
                # Print brief result
                if result["all_categories_correct"]:
                    status = "✓"
                elif result["any_category_correct"]:
                    status = "⚠"
                else:
                    status = "✗"
                    
                print(f"  {status} P:{result['precision']:.2f} R:{result['recall']:.2f} F1:{result['f1_score']:.2f}")
                
            except Exception as e:
                print(f"Error processing paper: {e}")
                # Add error record to results
                self.results.append({
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                    "error": str(e),
                    "success": False
                })
        
        # Calculate aggregate metrics
        successful_results = [r for r in self.results if "error" not in r]
        
        metrics = {
            "total_papers": len(benchmark_papers),
            "successful_calls": len(successful_results),
            "papers_with_all_categories_correct": sum(1 for r in successful_results if r.get("all_categories_correct", False)),
            "papers_with_any_category_correct": sum(1 for r in successful_results if r.get("any_category_correct", False)),
            "papers_with_any_correct_none_wrong": sum(1 for r in successful_results if r.get("any_correct_none_wrong", False)),
            "average_precision": sum(r.get("precision", 0) for r in successful_results) / len(successful_results) if successful_results else 0,
            "average_recall": sum(r.get("recall", 0) for r in successful_results) / len(successful_results) if successful_results else 0,
            "average_f1": sum(r.get("f1_score", 0) for r in successful_results) / len(successful_results) if successful_results else 0,
            "runtime_seconds": time.time() - start_time
        }
        
        # Print summary
        self._print_summary(metrics)
        
        # Return both detailed results and aggregated metrics
        return {
            "results": self.results,
            "metrics": metrics,
            "benchmark_config": {
                "csv_path": self.csv_path,
                "api_endpoint": self.api_endpoint,
                "sample_size": self.sample_size,
                "random_seed": self.random_seed,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _print_summary(self, metrics: Dict):
        """Print a summary of the benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total papers: {metrics['total_papers']}")
        print(f"Successful API calls: {metrics['successful_calls']}")
        print(f"Papers with all categories correct: {metrics['papers_with_all_categories_correct']} ({metrics['papers_with_all_categories_correct']/metrics['successful_calls']*100:.1f}%)")
        print(f"Papers with any category correct: {metrics['papers_with_any_category_correct']} ({metrics['papers_with_any_category_correct']/metrics['successful_calls']*100:.1f}%)")
        print(f"Papers with any correct & none wrong: {metrics['papers_with_any_correct_none_wrong']} ({metrics['papers_with_any_correct_none_wrong']/metrics['successful_calls']*100:.1f}%)")
        print(f"Average precision: {metrics['average_precision']:.4f}")
        print(f"Average recall: {metrics['average_recall']:.4f}")
        print(f"Average F1 score: {metrics['average_f1']:.4f}")
        print(f"Total runtime: {metrics['runtime_seconds']:.2f} seconds")
        print("="*60)
    
    def save_results(self, output_path: str):
        """Save benchmark results to a JSON file."""
        full_results = {
            "results": self.results,
            "metrics": {
                "total_papers": len(self.results),
                "papers_with_all_categories_correct": sum(1 for r in self.results if r.get("all_categories_correct", False)),
                "papers_with_any_category_correct": sum(1 for r in self.results if r.get("any_category_correct", False)),
                "papers_with_any_correct_none_wrong": sum(1 for r in self.results if r.get("any_correct_none_wrong", False)),
                "average_precision": sum(r.get("precision", 0) for r in self.results if "precision" in r) / len(self.results),
                "average_recall": sum(r.get("recall", 0) for r in self.results if "recall" in r) / len(self.results),
                "average_f1": sum(r.get("f1_score", 0) for r in self.results if "f1_score" in r) / len(self.results),
            },
            "benchmark_config": {
                "csv_path": self.csv_path,
                "api_endpoint": self.api_endpoint,
                "sample_size": self.sample_size,
                "random_seed": self.random_seed,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark for paper categorization model")
    parser.add_argument("--csv-path", help="Path to the CSV file with paper data", default="./data/cs_papers_api.csv")
    parser.add_argument("--api-endpoint", default="http://localhost:8000/api/queries/benchmark", help="API endpoint URL")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of papers to sample")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file path for results")
    
    args = parser.parse_args()
    
    benchmark = PaperCategorizationBenchmark(
        csv_path=args.csv_path,
        api_endpoint=args.api_endpoint,
        sample_size=args.sample_size,
        random_seed=args.random_seed
    )
    
    try:
        benchmark.run_benchmark()
        benchmark.save_results(args.output)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted. Saving partial results...")
        benchmark.save_results(args.output)
    except Exception as e:
        print(f"Error during benchmark execution: {e}")


if __name__ == "__main__":
    main()