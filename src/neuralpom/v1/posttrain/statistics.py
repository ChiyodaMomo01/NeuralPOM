"""Shared correction summaries; unchanged from the production trainer."""
import torch


def correction_statistics(records):
    """Equal-correction summaries; inventory sums retain their physical units."""
    metrics = {'correction/count': len(records)}
    for key in sorted({key for record in records for key in record}):
        values = torch.stack([torch.as_tensor(record[key]).detach() for record in records if key in record])
        for operation in ('sum', 'mean', 'min', 'max'):
            metrics[f'correction/{key}/{operation}'] = getattr(values, operation)()
    return metrics
