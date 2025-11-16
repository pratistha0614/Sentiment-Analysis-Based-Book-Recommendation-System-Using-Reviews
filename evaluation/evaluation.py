import csv
import json
import argparse

# Lightweight, decoupled evaluator.
# Does NOT import the Flask app to avoid heavy model/data loads.
# You supply a predictor callable that returns (label, confidence) for text.

CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
IDX = {c: i for i, c in enumerate(CLASSES)}


def confusion_matrix_3x3(y_true, y_pred):
    cm = [[0] * 3 for _ in range(3)]  # rows=true, cols=pred
    for t, p in zip(y_true, y_pred):
        if t in IDX and p in IDX:
            cm[IDX[t]][IDX[p]] += 1
    return cm


def per_class_metrics(cm):
    metrics = {}
    total = sum(sum(row) for row in cm)
    for i, cls in enumerate(CLASSES):
        TP = cm[i][i]
        FP = sum(cm[r][i] for r in range(3) if r != i)
        FN = sum(cm[i][c] for c in range(3) if c != i)
        TN = total - TP - FP - FN
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[cls] = dict(TP=TP, TN=TN, FP=FP, FN=FN,
                            precision=precision, recall=recall, f1=f1)
    return metrics


def overall_accuracy(cm):
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(3))
    return (correct / total) if total else 0.0


def load_csv_pairs(path):
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = row.get("text", "")
            lbl = (row.get("label", "").strip() or "").upper()
            if lbl in IDX:
                pairs.append((txt, lbl))
    return pairs


def evaluate_pairs(pairs, predictor):
    y_true, y_pred = [], []
    for text, true_label in pairs:
        pred_label, _ = predictor(text)
        y_true.append(true_label)
        y_pred.append(pred_label)
    cm = confusion_matrix_3x3(y_true, y_pred)
    per_cls = per_class_metrics(cm)
    acc = overall_accuracy(cm)
    macro_p = sum(per_cls[c]["precision"] for c in CLASSES) / 3
    macro_r = sum(per_cls[c]["recall"] for c in CLASSES) / 3
    macro_f = sum(per_cls[c]["f1"] for c in CLASSES) / 3
    return dict(
        confusion_matrix=cm,
        per_class=per_cls,
        accuracy=acc,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment predictor on CSV test set")
    parser.add_argument("--csv", required=True, help="Path to CSV with headers text,label")
    parser.add_argument("--output", default="-", help="Output JSON path or '-' for stdout")
    parser.add_argument("--demo", action="store_true", help="Use demo keyword predictor if no external predictor provided")
    args = parser.parse_args()

    pairs = load_csv_pairs(args.csv)
    if not pairs:
        raise SystemExit("No valid rows found in CSV (need headers: text,label)")

    # Demo keyword predictor (no heavy imports). Replace with real app call if needed.
    positive_keywords = {"excellent","amazing","great","love","loved","like","liked","wonderful","fantastic","awesome","brilliant","enjoyed","recommend","recommended","incredible","superb","favorite"}
    negative_keywords = {"bad","terrible","awful","hate","hated","dislike","disliked","boring","poor","waste","worst","disappointing","meh","not good","not great","confusing","predictable"}

    def keyword_predictor(text):
        t = (text or "").lower()
        pos = sum(1 for kw in positive_keywords if kw in t)
        neg = sum(1 for kw in negative_keywords if kw in t)
        if pos > neg:
            return ("POSITIVE", 1.0)
        elif neg > pos:
            return ("NEGATIVE", 1.0)
        else:
            return ("NEUTRAL", 1.0)

    predictor = keyword_predictor

    results = evaluate_pairs(pairs, predictor)
    if args.output == "-":
        print(json.dumps(results, indent=2))
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
