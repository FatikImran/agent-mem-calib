import json

def compute_f1(predictions: list[dict], golds: list[dict]):
    """
    Computes parameter-level precision, recall, and F1.
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for pred, gold in zip(predictions, golds):
        pred_args = pred.get("arguments", {}) if pred else {}
        gold_args = gold.get("arguments", {}) if gold else {}
        
        # Ensure they are dictionaries
        if not isinstance(pred_args, dict): pred_args = {}
        if not isinstance(gold_args, dict): gold_args = {}

        tp = 0
        for k, v in gold_args.items():
            if k in pred_args:
                # Check value (string matching, case-insensitive for strings)
                pred_v = pred_args[k]
                if str(v).lower() == str(pred_v).lower():
                    tp += 1
                    
        fp = len(pred_args) - tp
        fn = len(gold_args) - tp
        
        # Handle edge cases where fp or fn might be negative if model hallucinated keys
        fp = max(0, fp)
        fn = max(0, fn)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1