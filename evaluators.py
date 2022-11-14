import numpy as np
import pandas as pd

# -------------------------------------------------------
#   Classification evaluator
# -------------------------------------------------------
class ClassificationEvaluator():
    def compute_precision_recall():
        pass

# -------------------------------------------------------
#   Information retrieval evaluator
# -------------------------------------------------------
class IREvaluator():
    # Precision, Recall
    def compute_precision_recall_at_k(self, k=None, q_relevants={}, q_results=[]):
        if not k: k = len(q_results)
        num_correct = 0
        for q_result in q_results[:k]:
            if q_result['id'] in q_relevants: num_correct += 1 # hits
        return num_correct / k, num_correct / len(q_relevants)

    # Average Precision (note: Average Precision for single query, and Mean Average Precision for all query)
    def compute_avg_precision_at_k(self, k=None, q_relevants={}, q_results=[]):
        if not k: k = len(q_results)
        num_correct = 0
        sum_precisions = 0
        for rank, result in enumerate(q_results[:k]):
            if result['id'] in q_relevants:
                num_correct += 1
                sum_precisions += num_correct / (rank + 1)
        return sum_precisions / min(k, len(q_relevants))
    
    # Mean Reciprocal Ranking
    def compute_mrr_at_k(self, k=None, q_relevants={}, q_results=[]):
        if not k: k = len(q_results)
        for rank, q_result in enumerate(q_results[:k]):
            if q_result['id'] in q_relevants:
                return 1.0 / (rank + 1)
        return 0
    
    # Normalized Discounted Cumulative Gain
    def compute_ndcg_at_k(self, k=None, q_relevants={}, q_results=[]):
        if not k: k = len(q_results)
        predicted_relevances = [1 if q_result['id'] in q_relevants else 0 for q_result in q_results[:k]]
        true_relevances = [1] * len(q_relevants)
        ndcg = self.compute_dcg_at_k(predicted_relevances, k) / self.compute_dcg_at_k(true_relevances, k)
        return ndcg
    
    # Discounted Cumulative Gain
    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg

    def compute_metrics_at_ks(self, metrics=['precision', 'recall', 'map', 'mrr', 'ndcg'], ks=[1,5,10,20,50], relevants={}, results={}):
        # init score computation values
        scores = {
            'precision': {k: [] for k in ks},
            'recall': {k: [] for k in ks},
            'map': {k: [] for k in ks},
            'mrr': {k: [] for k in ks},
            'ndcg': {k: [] for k in ks}
        }

        # compute scores for all query search results
        for q, q_results in results.items():
            for k in ks:
                p, r = self.compute_precision_recall_at_k(k, relevants[q], q_results)
                scores['precision'][k].append(p)
                scores['recall'][k].append(r)

                avg_p = self.compute_avg_precision_at_k(k, relevants[q], q_results)
                scores['map'][k].append(avg_p)

                mrr = self.compute_mrr_at_k(k, relevants[q], q_results)
                scores['mrr'][k].append(mrr)

                ndcg = self.compute_ndcg_at_k(k, relevants[q], q_results)
                scores['ndcg'][k].append(ndcg)
        
        # compute averages
        for k in ks:
            scores['precision'][k] = np.mean(scores['precision'][k])
            scores['recall'][k] = np.mean(scores['recall'][k])
            scores['map'][k] = np.mean(scores['map'][k])
            scores['mrr'][k] = np.mean(scores['mrr'][k])
            scores['ndcg'][k] = np.mean(scores['ndcg'][k])

        # output
        return {m: s for m, s in scores.items() if m in metrics}


if __name__ == '__main__':
    ir_evaluator = IREvaluator()
    metrics = ir_evaluator.compute_metrics_at_ks(
        metrics=['precision', 'recall', 'map', 'mrr', 'ndcg'],
        ks=[1, 5, 10, 20, 50],
        relevants={
            1: {2, 3, 4},
            2: {3, 5},
            3: {4, 6}
        },
        results={
            1: [
                {'id': 3, 'score': 0.9},
                {'id': 2, 'score': 0.8},
                {'id': 4, 'score': 0.7},
            ],
            2: [
                {'id': 2, 'score': 0.9},
                {'id': 3, 'score': 0.8},
                {'id': 5, 'score': 0.7},
            ],
            3: [
                {'id': 4, 'score': 0.9},
                {'id': 6, 'score': 0.8},
                {'id': 5, 'score': 0.7},
            ],
        }
    )
    print(pd.DataFrame(metrics).T)