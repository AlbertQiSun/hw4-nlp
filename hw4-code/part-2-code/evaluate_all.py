#!/usr/bin/env python3
"""
Script to evaluate all T5 models and show their metrics
"""
import os
from utils import compute_metrics

def evaluate_model(model_name, sql_file, records_file):
    """Evaluate a model and return its metrics"""
    gt_sql = 'data/dev.sql'
    gt_records = 'records/ground_truth_dev.pkl'

    if not os.path.exists(sql_file):
        print(f"Warning: {sql_file} not found, skipping {model_name}")
        return None

    if not os.path.exists(records_file):
        print(f"Warning: {records_file} not found, skipping {model_name}")
        return None

    sql_em, record_em, record_f1, error_msgs = compute_metrics(gt_sql, sql_file, gt_records, records_file)

    error_rate = sum(1 for msg in error_msgs if msg) / len(error_msgs) if error_msgs else 0

    return {
        'model': model_name,
        'sql_em': sql_em,
        'record_em': record_em,
        'record_f1': record_f1,
        'error_rate': error_rate
    }

def main():
    print("Evaluating T5 models on development set...")
    print("=" * 60)

    # Evaluate fine-tuned model
    ft_results = evaluate_model(
        "Fine-tuned T5",
        "results/t5_ft_ft_experiment_dev.sql",
        "records/t5_ft_ft_experiment_dev.pkl"
    )

    # Evaluate from-scratch model (if results exist)
    scr_results = evaluate_model(
        "T5 from scratch",
        "results/t5_scr_test.sql",  # This might be test set, let me check
        "records/t5_scr_test.pkl"
    )

    # Also check if there's a dev set result for scratch model
    scr_dev_results = evaluate_model(
        "T5 from scratch (dev)",
        "results/t5_scr_scr_experiment_dev.sql",
        "records/t5_scr_scr_experiment_dev.pkl"
    )

    print("\nResults:")
    print("-" * 40)

    for results in [ft_results, scr_dev_results, scr_results]:
        if results:
            print(f"\n{results['model']}:")
            print(f"  SQL EM:     {results['sql_em']:.4f}")
            print(f"  Record EM:  {results['record_em']:.4f}")
            print(f"  Record F1:  {results['record_f1']:.4f}")
            print(f"  Error Rate: {results['error_rate']:.2%}")

if __name__ == "__main__":
    main()

