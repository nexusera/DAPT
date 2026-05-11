import argparse
import json
import sys
import os
import logging
import datetime
from collections import defaultdict

# Add project root to system path to ensure med_eval can be imported
sys.path.insert(0, os.getcwd())

# Import evaluation orchestrator
from med_eval.engine import run_evaluation

# Configure global logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_jsonl(filepath):
    """
    Load JSONL formatted files.
    Parses line by line and handles potential JSON errors or empty lines.
    """
    data = []
    if not filepath or not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"Line parsing failed: {line[:50]}... Error: {e}")
    return data

def load_query_set(p):
    """
    Load key-alias mapping (Query Set).
    Used for alias alignment and standard field list determination in Task 2.
    """
    if not p or not os.path.exists(p):
        logger.warning(f"Query Set file not found, Task 2 may be limited: {p}")
        return {}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load Query Set: {e}")
        return {}


def main():
    """
    Main evaluation entry point:
    1. Parse CLI arguments and configure engine behavior.
    2. Load prediction and ground truth files.
    3. Execute standardized conversion and task-specific filtering.
    4. Invoke engine for metrics and generate a standardized JSON report.
    """
    parser = argparse.ArgumentParser(description="Medical Structured Data Evaluation Tool (Unified Modular Scorer)")
    
    # Basic file I/O parameters
    parser.add_argument("--pred_file", required=True, help="Model prediction file (.jsonl)")
    parser.add_argument("--gt_file", required=True, help="Ground truth file (.jsonl)")
    parser.add_argument("--query_set", dest="query_set_file", default="<path_to_query_set.json>", help="Alias mapping and standard field set (Query Set)")
    parser.add_argument("--output_file", default=None, help="Output path for results JSON")
    parser.add_argument("--task_type", default="all", choices=["task1", "task2", "task3", "all"], help="Run specific task evaluation")
    
    # Algorithm behavior control: normalization, dynamic threshold, and position verification
    parser.add_argument("--no_normalize", action="store_false", dest="normalize", help="Disable text normalization (lowercase, whitespace removal)")
    parser.add_argument("--similarity_threshold", type=float, default=0.8, help="NED similarity threshold")
    parser.add_argument("--overlap_threshold", type=float, default=0.0, help="Span IoU overlap threshold")
    parser.add_argument("--disable_tau", action="store_false", dest="tau_dynamic", help="Disable Tau dynamic threshold")
    
    # Metadata info (used for summary report only)
    parser.add_argument("--model_name", default=None, help="Model name identifier")
    parser.add_argument("--dataset_type", default="MedStruct-S", help="Dataset type identifier (MedStruct-S / MedStruct-S (De-identified))")
    
    parser.set_defaults(normalize=True, tau_dynamic=True)
    args = parser.parse_args()

    # Load IO
    logger.info(f"Loading prediction file: {args.pred_file}")
    predictions = load_jsonl(args.pred_file)
    logger.info(f"Loading ground truth file: {args.gt_file}")
    ground_truth = load_jsonl(args.gt_file)
    
    # Assert sample counts match
    if len(predictions) != len(ground_truth):
        logger.error(f"Sample count mismatch: Preds={len(predictions)}, GT={len(ground_truth)}")
        sys.exit(1)
        
    num_samples = len(predictions)
    logger.info(f"Processing {num_samples} samples for comparison.")
    
    # Load Query Set
    query_set = load_query_set(args.query_set_file)
    
    # Config object for reporting only
    report_config = {
        "normalize": args.normalize,
        "similarity_threshold": args.similarity_threshold,
        "overlap_threshold": args.overlap_threshold,
        "tau_dynamic": args.tau_dynamic,
        "use_em": True,
        "use_am": True,
        "use_span": True
    }

    # Execute
    logger.info(f"Starting task evaluation: {args.task_type}...")
    
    results = run_evaluation(
        predictions=predictions,
        ground_truth=ground_truth,
        query_set=query_set,
        task_type=args.task_type,
        normalize=args.normalize,
        tau_dynamic=args.tau_dynamic,
        similarity_threshold=args.similarity_threshold,
        overlap_threshold=args.overlap_threshold
    )

    # Assemble final standardized report
    final_report = {
        "summary": {
            "model": args.model_name or os.path.basename(args.pred_file),
            "dataset": args.dataset_type,
            "samples": num_samples,
            "eval_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": report_config
        },
        "tasks": {}
    }

    # Task name mapping for JSON keys
    task_map = {
        "Task 1 (Key Discovery)": "task1",
        "Task 2 (Value Extraction)": "task2",
        "Task 3 (E2E Pairing)": "task3"
    }

    # Flattening logic: Lift Task 2 dimensions to top-level tasks
    for raw_name, data in results.items():
        clean_key = task_map.get(raw_name, raw_name.lower().replace(" ", "_"))
        if clean_key == "task2":
            # Split Task 2 into Global/Pos-only dimensions for easier parsing
            final_report["tasks"]["task2_global"] = data["global"]
            final_report["tasks"]["task2_pos_only"] = data["pos_only"]
        else:
            final_report["tasks"][clean_key] = data

    # Serialize to JSON and print
    report_json = json.dumps(final_report, indent=2, ensure_ascii=False)
    print(report_json)
    
    # Persist results if output path is specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(report_json)
        logger.info(f"Report saved to: {args.output_file}")

if __name__ == "__main__":
    main()
