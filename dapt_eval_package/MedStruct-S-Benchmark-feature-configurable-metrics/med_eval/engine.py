from med_eval.evaluators.task1_evaluator import evaluate_task1_discovery
from med_eval.evaluators.task2_evaluator import evaluate_task2_qa
from med_eval.evaluators.task3_evaluator import evaluate_task3_pairing

def run_evaluation(predictions, ground_truth, key_alias_map=None, task_type="all", normalize=True, tau_dynamic=True, similarity_threshold=0.8, overlap_threshold=0.0):
    """
    Main entry point for running one or more task evaluations.
    
    Args:
        predictions: Standardized prediction samples.
        ground_truth: Standardized GT samples.
        key_alias_map: Schema mapping (required for Task 2).
        task_type: 'task1', 'task2', 'task3', or 'all'.
        normalize: Whether to normalize text before comparison.
        tau_dynamic: Whether to use dynamic tau for fuzzy matching.
        similarity_threshold: Threshold for fuzzy string matching.
        overlap_threshold: Threshold for overlap-based matching.
        
    Returns:
        Dictionary containing results for requested tasks.
    """
    results = {}
    
    # Task mapping
    task_map = {
        "task1": evaluate_task1_discovery,
        "task2": evaluate_task2_qa,
        "task3": evaluate_task3_pairing
    }
    
    # Determine which tasks to run
    # Directly execute requested tasks
    if task_type in ["all", "task1"]:
        results["Task 1 (Key Discovery)"] = evaluate_task1_discovery(
            predictions, 
            ground_truth, 
            normalize=normalize,
            tau_dynamic=tau_dynamic,
            similarity_threshold=similarity_threshold,
            overlap_threshold=overlap_threshold
        )
        
    if task_type in ["all", "task2"]:
        if not key_alias_map:
            raise ValueError("Task 2 requires 'key_alias_map' (Schema).")
        results["Task 2 (Value Extraction)"] = evaluate_task2_qa(
            predictions, 
            ground_truth, 
            key_alias_map,
            normalize=normalize,
            tau_dynamic=tau_dynamic,
            similarity_threshold=similarity_threshold
        )
        
    if task_type in ["all", "task3"]:
        results["Task 3 (E2E Pairing)"] = evaluate_task3_pairing(
            predictions, 
            ground_truth, 
            normalize=normalize,
            tau_dynamic=tau_dynamic,
            similarity_threshold=similarity_threshold,
            overlap_threshold=overlap_threshold
        )
        
    return results
