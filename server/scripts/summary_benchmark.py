import os
import json
import random
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
from pypdf import PdfReader
import time
from typing import List, Dict, Any, Tuple
from openai import OpenAI

# Configuration
API_ENDPOINT = "http://localhost:8000/api/queries/benchmark"
DATASET_DIR = "./data/papers/"
OUTPUT_DIR = "evaluation_results"
NUM_SAMPLES = 14  # Number of papers to evaluate

# OpenAI API Configuration for LLM-as-Judge
OPENAI_BASE_URL = "https://nerc.guha-anderson.com/v1"
OPENAI_API_KEY = "zheng.kaiy@northeastern.edu:39258"
OPENAI_MODEL = "llama3p1-8b-instruct"  # or another appropriate model
JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator of AI paper query responses. 
Given a query about an academic paper and two responses (A and B), your task is to evaluate which response is better.

Paper content excerpt: 
{paper_excerpt}

Query: {query}

Response A (System 1): 
{response_a}

Response B (System 2): 
{response_b}

Please evaluate the responses based on the following criteria:
1. Relevance: How well does the response address the query?
2. Accuracy: Is the information provided factually correct based on the paper content?
3. Completeness: Does the response provide a comprehensive answer to the query?
4. Clarity: Is the response clear, well-organized, and easy to understand?

For each criterion, assign a score from 1-5, where 5 is excellent and 1 is poor.
Also provide a final verdict on which response is better overall (A, B, or Tie) with a brief explanation.

Output your evaluation in the following JSON format:
{{
    "relevance_a": score,
    "relevance_b": score,
    "accuracy_a": score,
    "accuracy_b": score,
    "completeness_a": score,
    "completeness_b": score,
    "clarity_a": score,
    "clarity_b": score,
    "better_response": "A" or "B" or "Tie",
    "explanation": "your explanation here"
}}
"""

# Generate query templates for papers
QUERY_TEMPLATES = [
    "What are the main contributions of this paper?",
    "What methods does this paper use?",
    "How does this paper compare to related work?",
    "What are the limitations of this work?",
    "What future work does this paper suggest?",
    "Can you summarize the experimental results?",
    "What datasets were used in this paper?",
    "What are the key findings of this paper?",
    "What problem is this paper trying to solve?",
    "what papers are related to this one?"
]

def ensure_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path: str, max_pages: int = 5) -> str:
    """Extract text from a PDF file, limiting to first few pages for efficiency."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i in range(min(len(reader.pages), max_pages)):
            text += reader.pages[i].extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def get_paper_sample(dataset_dir: str, n_samples: int) -> List[Dict[str, Any]]:
    """Get a random sample of papers from the dataset."""
    all_pdfs = [f for f in os.listdir(dataset_dir) if f.endswith('.pdf')]
    if n_samples > len(all_pdfs):
        n_samples = len(all_pdfs)
    
    selected_pdfs = random.sample(all_pdfs, n_samples)
    papers = []
    
    for pdf_file in tqdm(selected_pdfs, desc="Loading papers"):
        pdf_path = os.path.join(dataset_dir, pdf_file)
        paper_text = extract_text_from_pdf(pdf_path)
        if paper_text:  # Only include if text extraction succeeded
            papers.append({
                "file_name": pdf_file,
                "paper_content": paper_text,
                "query": random.choice(QUERY_TEMPLATES)
            })
    
    return papers

def query_api(paper: Dict[str, Any], use_rag: bool) -> Dict[str, Any]:
    """Query the API with or without RAG."""
    payload = {
        "prompt": paper["query"],
        "paper_content": paper["paper_content"],
        "use_rag": use_rag
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error querying API for {paper['file_name']}: {e}")
        return None

def evaluate_with_llm_judge(paper_excerpt: str, query: str, response_a: str, response_b: str) -> Dict[str, Any]:
    """Use OpenAI's model as a judge to evaluate two responses."""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        paper_excerpt=paper_excerpt[:2000],  # Limit size for efficiency
        query=query,
        response_a=response_a,
        response_b=response_b
    )
    
    try:
        # Initialize OpenAI client
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert academic evaluator tasked with comparing responses about research papers."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0  # Using a low temperature for more consistent evaluations
        )
        
        # Extract and parse the JSON response
        result = response.choices[0].message.content
        try:
            # Try to parse the JSON from the response
            return json.loads(result)
        except json.JSONDecodeError:
            # If response is not valid JSON, log the issue and try to extract JSON portion
            print(f"Invalid JSON received from LLM judge. Attempting to extract JSON portion...")
            # Look for JSON-like structure within the response
            import re
            json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
            match = re.search(json_pattern, result)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    print(f"Failed to extract valid JSON. Raw response: {result[:200]}...")
                    return None
            print(f"No JSON structure found. Raw response: {result[:200]}...")
            return None
    except Exception as e:
        print(f"Error with OpenAI LLM judge: {e}")
        return None

def process_paper(paper: Dict[str, Any], randomize_order: bool = True) -> Dict[str, Any]:
    """Process a single paper through both systems and evaluate results."""
    print(f"Processing paper: {paper['file_name']}")
    
    # Get responses from both systems
    try:
        rag_response = query_api(paper, use_rag=True)
        non_rag_response = query_api(paper, use_rag=False)
        
        if not rag_response or not non_rag_response:
            print(f"  Failed to get responses for {paper['file_name']}")
            return None
        
        # Randomize the order to avoid position bias in the judge
        if randomize_order and random.random() < 0.5:
            response_a, response_b = rag_response["response"], non_rag_response["response"]
            system_a, system_b = "RAG", "Non-RAG"
        else:
            response_a, response_b = non_rag_response["response"], rag_response["response"]
            system_a, system_b = "Non-RAG", "RAG"
        
        # Get evaluation from LLM judge
        judge_result = evaluate_with_llm_judge(
            paper["paper_content"], 
            paper["query"],
            response_a,
            response_b
        )
        
        if not judge_result:
            print(f"  Failed to get evaluation for {paper['file_name']}")
            return None
            
        # Map the results back to RAG and Non-RAG
        result = {
            "file_name": paper["file_name"],
            "query": paper["query"],
            "rag_response": rag_response["response"],
            "non_rag_response": non_rag_response["response"],
            "raw_judge_result": str(judge_result),  # Convert to string to avoid circular references
            "system_a": system_a,
            "system_b": system_b
        }
        
        # Remap the scores to RAG and Non-RAG
        metrics = ["relevance", "accuracy", "completeness", "clarity"]
        for metric in metrics:
            # Handle potential missing metrics in judge result
            metric_a = judge_result.get(f"{metric}_a", 3)  # Default to 3 if missing
            metric_b = judge_result.get(f"{metric}_b", 3)  # Default to 3 if missing
            
            # Ensure these are primitive types
            if not isinstance(metric_a, (int, float)):
                metric_a = 3
            if not isinstance(metric_b, (int, float)):
                metric_b = 3
                
            result[f"{metric}_rag"] = float(metric_b if system_b == "RAG" else metric_a)
            result[f"{metric}_non_rag"] = float(metric_a if system_b == "RAG" else metric_b)
        
        # Determine winner
        better_response = judge_result.get("better_response", "Tie")
        if better_response not in ["A", "B", "Tie"]:
            better_response = "Tie"
            
        if better_response == "Tie":
            result["winner"] = "Tie"
        else:
            is_rag_winner = (better_response == "A" and system_a == "RAG") or \
                            (better_response == "B" and system_b == "RAG")
            result["winner"] = "RAG" if is_rag_winner else "Non-RAG"
        
        return result
    except Exception as e:
        print(f"  Error processing {paper['file_name']}: {e}")
        return None

def run_evaluation(dataset_dir: str, n_samples: int) -> List[Dict[str, Any]]:
    """Run the evaluation process without plotting or HTML generation."""
    ensure_dirs()
    papers = get_paper_sample(dataset_dir, n_samples)
    print(f"Evaluating {len(papers)} papers...")
    
    results = []
    for paper in tqdm(papers, desc="Evaluating"):
        result = process_paper(paper)
        if result:
            results.append(result)
    
    # Save raw results
    with open(os.path.join(OUTPUT_DIR, "raw_results.json"), "w") as f:
        # Convert any non-serializable objects to strings
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable_result[key] = value
                else:
                    serializable_result[key] = str(value)
            serializable_results.append(serializable_result)
        json.dump(serializable_results, f, indent=2)
    
    return results

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the evaluation results without plots or HTML generation."""
    if not results:
        return {"error": "No valid results to analyze"}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Overall winner counts
    winner_counts = df["winner"].value_counts().to_dict()
    win_percentage = {
        "RAG": winner_counts.get("RAG", 0) / len(df) * 100,
        "Non-RAG": winner_counts.get("Non-RAG", 0) / len(df) * 100,
        "Tie": winner_counts.get("Tie", 0) / len(df) * 100
    }
    
    # Average scores per metric
    metrics = ["relevance", "accuracy", "completeness", "clarity"]
    avg_scores = {
        "RAG": {metric: float(df[f"{metric}_rag"].mean()) for metric in metrics},
        "Non-RAG": {metric: float(df[f"{metric}_non_rag"].mean()) for metric in metrics}
    }
    
    # Score differences (RAG - Non-RAG)
    score_diffs = {metric: df[f"{metric}_rag"] - df[f"{metric}_non_rag"] for metric in metrics}
    avg_score_diffs = {metric: float(score_diffs[metric].mean()) for metric in metrics}
    
    # Determine which system performed better for each metric
    better_system = {metric: "RAG" if avg_score_diffs[metric] > 0 else "Non-RAG" if avg_score_diffs[metric] < 0 else "Tie" 
                     for metric in metrics}
    
    # Statistical significance
    from scipy import stats
    p_values = {}
    for metric in metrics:
        t_stat, p_val = stats.ttest_rel(df[f"{metric}_rag"], df[f"{metric}_non_rag"])
        p_values[metric] = float(p_val)
    
    # Generate overall better system
    overall_winner = "RAG" if winner_counts.get("RAG", 0) > winner_counts.get("Non-RAG", 0) else "Non-RAG" if winner_counts.get("Non-RAG", 0) > winner_counts.get("RAG", 0) else "Tie"
    
    # Ensure all values are JSON serializable
    analysis = {
        "total_samples": int(len(df)),
        "winner_counts": {k: int(v) for k, v in winner_counts.items()},
        "win_percentage": {k: float(v) for k, v in win_percentage.items()},
        "average_scores": avg_scores,
        "score_differences": avg_score_diffs,
        "better_system": better_system,
        "overall_winner": overall_winner,
        "p_values": p_values,
        "is_significant": {metric: bool(p_val < 0.05) for metric, p_val in p_values.items()}
    }
    
    # Save analysis
    try:
        with open(os.path.join(OUTPUT_DIR, "analysis_results.json"), "w") as f:
            json.dump(analysis, f, indent=2)
    except (TypeError, ValueError) as e:
        print(f"Warning: Could not save analysis results to JSON: {e}")
        # Save a simplified version
        simple_analysis = {
            "total_samples": analysis["total_samples"],
            "winner_counts": analysis["winner_counts"],
            "overall_winner": analysis["overall_winner"],
            "better_system": analysis["better_system"]
        }
        with open(os.path.join(OUTPUT_DIR, "simplified_analysis_results.json"), "w") as f:
            json.dump(simple_analysis, f, indent=2)
    
    return analysis

def create_comparison_plots(df: pd.DataFrame, metrics: List[str]):
    """Placeholder function - plotting removed per user request."""
    # All plotting code removed
    pass

def generate_html_report(df: pd.DataFrame, metrics: List[str]):
    """Placeholder function - HTML generation removed per user request."""
    # All HTML generation code removed
    pass

def test_openai_connection():
    """Test the OpenAI API connection before running the full evaluation."""
    print("Testing OpenAI API connection...")
    try:
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Respond with 'Connection successful' if you receive this message."}
            ],
            max_tokens=20
        )
        print(f"OpenAI API test response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Error connecting to OpenAI API: {e}")
        return False

def main():
    """Main function to run the evaluation."""
    print("Starting RAG vs Non-RAG evaluation...")
    start_time = time.time()
    
    # Test OpenAI connection first
    if not test_openai_connection():
        print("Failed to connect to OpenAI API. Please check your credentials and try again.")
        return
    
    results = run_evaluation(DATASET_DIR, NUM_SAMPLES)
    analysis = analyze_results(results)
    
    if "error" in analysis:
        print(f"Error in analysis: {analysis['error']}")
        return
    
    print("\n===== EVALUATION RESULTS =====")
    print(f"Total samples evaluated: {analysis['total_samples']}")
    print(f"Overall winner: {analysis['overall_winner']}")
    print(f"Win counts: {analysis['winner_counts']}")
    print(f"Win percentages: RAG: {analysis['win_percentage']['RAG']:.1f}%, "
          f"Non-RAG: {analysis['win_percentage']['Non-RAG']:.1f}%, "
          f"Tie: {analysis['win_percentage']['Tie']:.1f}%")
    
    print("\nBetter system for each metric:")
    for metric, better in analysis["better_system"].items():
        diff = analysis["score_differences"][metric]
        significance = "significant" if analysis["is_significant"][metric] else "not significant"
        print(f"  {metric.capitalize()}: {better} is better by {abs(diff):.2f} points ({significance}, p={analysis['p_values'][metric]:.4f})")
    
    print("\nAverage scores:")
    for system in ["RAG", "Non-RAG"]:
        print(f"  {system}:")
        for metric, score in analysis["average_scores"][system].items():
            print(f"    {metric.capitalize()}: {score:.2f}/5")
    
    print(f"\nDetailed results saved to {OUTPUT_DIR}/")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()