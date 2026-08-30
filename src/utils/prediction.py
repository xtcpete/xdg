import torch


def vote_ensemble_probs(probs_a, probs_b):
    vote_0 = (probs_a[:, 0] > probs_a[:, 1]).long() + (probs_b[:, 0] > probs_b[:, 1]).long()
    vote_1 = (probs_a[:, 1] > probs_a[:, 0]).long() + (probs_b[:, 1] > probs_b[:, 0]).long()

    positive_scores = torch.stack([probs_a[:, 1], probs_b[:, 1]], dim=1)
    final_positive = positive_scores.mean(dim=1)
    final_positive = torch.where(
        vote_1 > vote_0,
        positive_scores.max(dim=1).values,
        final_positive,
    )
    final_positive = torch.where(
        vote_1 < vote_0,
        positive_scores.min(dim=1).values,
        final_positive,
    )
    return torch.stack([1.0 - final_positive, final_positive], dim=1)


def vote_ensemble_logits(logits_a, logits_b):
    probs_a = torch.softmax(logits_a, dim=1)
    probs_b = torch.softmax(logits_b, dim=1)
    merged_probs = vote_ensemble_probs(probs_a, probs_b)
    return merged_probs.clamp_min(torch.finfo(merged_probs.dtype).tiny).log()


def vote_symmetrical_logits(logits):
    if logits.shape[0] % 2 != 0:
        raise ValueError(
            f"Expected an even number of symmetric logits, got {logits.shape[0]}."
        )
    half = logits.shape[0] // 2
    return vote_ensemble_logits(logits[:half], logits[half:])
