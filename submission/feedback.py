import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from submission.corpus_utils import load_corpus
from submission.lm_utils import CollectionStats, dirichlet_smoothed_log_prob, tokenize

# Optimized Parameters for RM3
DIRICHLET_MU = 900.0
MAX_EXPANSION_TERMS = 12
CUMULATIVE_MASS_CUTOFF = 0.75

_STATS: Optional[CollectionStats] = None


def prepare(corpus_path: str) -> None:
    """Load the corpus and build collection-wide statistics once."""
    global _STATS
    corpus = load_corpus(corpus_path)
    _STATS = CollectionStats.from_corpus(corpus)


def score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """Dirichlet-Smoothed Unigram Query Likelihood baseline."""
    if _STATS is None:
        raise RuntimeError("score_candidates() called before prepare()")
    return _ql_rerank(query, candidate_doc_ids, k, _STATS)


def relevance_model_feedback(
    query: str,
    pseudo_relevant_doc_ids: List[str],
    candidate_doc_ids: List[str],
    k: int = 10,
) -> List[Tuple[str, float]]:
    """Noise-Resilient RM3 Relevance Model with Log-Sum-Exp Stability."""
    if _STATS is None:
        raise RuntimeError("relevance_model_feedback() called before prepare()")

    query_terms = tokenize(query)
    if not query_terms or not candidate_doc_ids:
        return []

    if not pseudo_relevant_doc_ids:
        return _ql_rerank(query, candidate_doc_ids, k, _STATS)

    # -------------------------------------------------------------
    # Step 1: Compute P(d|q) with Log-Sum-Exp Trick for Stability
    # -------------------------------------------------------------
    log_p_q_given_d: Dict[str, float] = {}
    
    for doc_id in pseudo_relevant_doc_ids:
        doc_len = _STATS.doc_lengths.get(doc_id, 0)
        if doc_len == 0:
            continue

        term_counts = _STATS.doc_term_counts(doc_id)
        log_p_q_d = 0.0
        for q_term in query_terms:
            p_coll = _STATS.collection_prob(q_term)
            if p_coll <= 0:
                continue
            log_p_q_d += dirichlet_smoothed_log_prob(
                term_counts.get(q_term, 0), doc_len, p_coll, DIRICHLET_MU
            )

        log_p_q_given_d[doc_id] = log_p_q_d

    if not log_p_q_given_d:
        return _ql_rerank(query, candidate_doc_ids, k, _STATS)

    # Prevent floating-point underflow via Log-Sum-Exp
    max_log_prob = max(log_p_q_given_d.values())
    p_q_given_d = {doc_id: math.exp(lp - max_log_prob) for doc_id, lp in log_p_q_given_d.items()}
    total_p_q_d = sum(p_q_given_d.values())

    p_d_given_q = {doc_id: prob / total_p_q_d for doc_id, prob in p_q_given_d.items()}

    # -------------------------------------------------------------
    # Step 2: Standard Relevance Model Term Weighting P(w|R)
    # -------------------------------------------------------------
    p_w_r: Dict[str, float] = defaultdict(float)

    for doc_id, doc_p_d in p_d_given_q.items():
        if doc_p_d <= 0:
            continue

        doc_len = _STATS.doc_lengths.get(doc_id, 0)
        term_counts = _STATS.doc_term_counts(doc_id)

        for term, count in term_counts.items():
            p_coll = _STATS.collection_prob(term)
            if p_coll <= 0:
                continue

            # Standard Dirichlet Smoothed term probability in doc
            p_w_d = (count + DIRICHLET_MU * p_coll) / (doc_len + DIRICHLET_MU)
            p_w_r[term] += p_w_d * doc_p_d

    total_weight = sum(p_w_r.values())
    if total_weight <= 0:
        return _ql_rerank(query, candidate_doc_ids, k, _STATS)

    for term in p_w_r:
        p_w_r[term] /= total_weight

    # -------------------------------------------------------------
    # Step 3: Mass Cutoff Truncation
    # -------------------------------------------------------------
    sorted_terms = sorted(p_w_r.items(), key=lambda x: x[1], reverse=True)

    p_rm_truncated: Dict[str, float] = {}
    accumulated_mass = 0.0

    for term, weight in sorted_terms[:MAX_EXPANSION_TERMS]:
        p_rm_truncated[term] = weight
        accumulated_mass += weight
        if accumulated_mass >= CUMULATIVE_MASS_CUTOFF:
            break

    trunc_sum = sum(p_rm_truncated.values())
    if trunc_sum > 0:
        for t in p_rm_truncated:
            p_rm_truncated[t] /= trunc_sum

    # -------------------------------------------------------------
    # Step 4: Query-Anchored RM3 Interpolation
    # -------------------------------------------------------------
    p_final: Dict[str, float] = defaultdict(float)
    q_counts = Counter(query_terms)
    q_len = len(query_terms)

    adaptive_lambda = min(0.75, max(0.65, 0.55 + 0.03 * q_len))

    for term, count in q_counts.items():
        p_final[term] += adaptive_lambda * (count / q_len)

    for term, weight in p_rm_truncated.items():
        p_final[term] += (1.0 - adaptive_lambda) * weight

    # -------------------------------------------------------------
    # Step 5: Candidate Pool Reranking
    # -------------------------------------------------------------
    scores: Dict[str, float] = {}
    for doc_id in candidate_doc_ids:
        doc_len = _STATS.doc_lengths.get(doc_id, 0)
        if doc_len == 0:
            scores[doc_id] = -1e9
            continue

        term_counts = _STATS.doc_term_counts(doc_id)
        doc_score = 0.0

        for w, weight in p_final.items():
            if weight <= 0:
                continue
            p_coll = _STATS.collection_prob(w)
            if p_coll <= 0:
                continue  # Guard against missing terms

            log_p_w_d = dirichlet_smoothed_log_prob(
                term_counts.get(w, 0), doc_len, p_coll, DIRICHLET_MU
            )
            doc_score += weight * log_p_w_d

        scores[doc_id] = doc_score

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


def _ql_rerank(query: str, doc_ids: List[str], k: int, stats: CollectionStats) -> List[Tuple[str, float]]:
    """Dirichlet-smoothed Unigram Query-Likelihood Reranker."""
    query_terms = tokenize(query)
    if not query_terms:
        return []

    scores: Dict[str, float] = {}
    for doc_id in doc_ids:
        doc_length = stats.doc_lengths.get(doc_id, 0)
        if doc_length == 0:
            scores[doc_id] = -1e9
            continue

        term_counts = stats.doc_term_counts(doc_id)
        log_prob = 0.0
        for term in query_terms:
            p_collection = stats.collection_prob(term)
            if p_collection <= 0:
                continue
            log_prob += dirichlet_smoothed_log_prob(
                term_counts.get(term, 0), doc_length, p_collection, DIRICHLET_MU
            )
        scores[doc_id] = log_prob

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]