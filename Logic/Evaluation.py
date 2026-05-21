import math
from typing import List


class Evaluation:
    def __init__(self, name: str):
        self.name = name

    def _validate(self, actual: List[List[str]], predicted: List[List[str]]):
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have the same length")

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro precision.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0

        total_precision = 0.0
        for act, pred in zip(actual, predicted):
            if not pred:
                continue
            true_positives = len(set(act) & set(pred))
            total_precision += true_positives/len(pred)

        return total_precision / len(actual)

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro recall.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0

        total_recall = 0.0
        for act, pred in zip(actual, predicted):
            if not act:
                continue
            true_positives = len(set(act) & set(pred))
            total_recall += true_positives / len(act)

        return total_recall / len(actual)

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates F1 score.
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)

        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _average_precision_single(self, actual: List[str], predicted: List[str]) -> float:
        if not actual or not predicted:
            return 0.0

        hits = 0
        sum_predictions = 0.0
        for i, doc_id in enumerate(predicted):
            if doc_id in actual:
                hits += 1
                sum_predictions += hits / (i+1)

        if len(actual) == 0:
            return 0.0
        return sum_predictions / len(actual)

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean AP across all queries.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0

        total_ap = sum(self._average_precision_single(act, pred) for act, pred in zip(actual, predicted))
        return total_ap / len(actual)

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MAP.
        """
        return self.calculate_AP(actual, predicted)

    def _dcg_single(self, actual: List[str], predicted: List[str]) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(predicted):
            if doc_id in actual:
                # rank index is i + 1
                dcg += 1.0 / math.log2(i + 2)
        return dcg

    def calculate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean DCG.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0

        total_dcg = sum(self._dcg_single(act, pred) for act, pred in zip(actual, predicted))
        return total_dcg / len(actual)

    def calculate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean NDCG.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0

        total_ndcg = 0.0
        for act, pred in zip(actual, predicted):
            dcg = self._dcg_single(act, pred)

            # calculate idcg
            idcg = 0.0
            for i in range(min(len(act), len(pred))):
                idcg += 1.0 / math.log2(i + 2)

            if idcg > 0:
                total_ndcg += dcg / idcg

        return total_ndcg / len(actual)

    def calculate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculate reciprocal rank.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0

        total_rr = 0.0
        for act, pred in zip(actual, predicted):
            for i, doc_id in enumerate(pred):
                if doc_id in act:
                    total_rr += 1.0 / (i + 1)
                    break  # stop checking other ranks when finding the first relevant doc

        return total_rr / len(actual)


    def calculate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MRR.
        """
        return self.calculate_RR(actual, predicted)

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics.
        """
        print(f"name = {self.name}")
        print(f"Precision = {precision:.6f}")
        print(f"Recall = {recall:.6f}")
        print(f"F1 = {f1:.6f}")
        print(f"AP = {ap:.6f}")
        print(f"MAP = {map:.6f}")
        print(f"DCG = {dcg:.6f}")
        print(f"NDCG = {ndcg:.6f}")
        print(f"RR = {rr:.6f}")
        print(f"MRR = {mrr:.6f}")

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics.
        """
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'ap': ap,
                    'map': map,
                    'dcg': dcg,
                    'ndcg': ndcg,
                    'rr': rr,
                    'mrr': mrr,
                })
        except Exception:
            pass

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        Call all functions to calculate evaluation metrics.
        """
        self._validate(actual, predicted)

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_val = self.calculate_MAP(actual, predicted)
        dcg = self.calculate_DCG(actual, predicted)
        ndcg = self.calculate_NDCG(actual, predicted)
        rr = self.calculate_RR(actual, predicted)
        mrr = self.calculate_MRR(actual, predicted)

        self.print_evaluation(precision, recall, f1, ap, map_val, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_val, dcg, ndcg, rr, mrr)
