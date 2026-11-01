"""Four predeclared full-epoch endpoints: 3, 6, 12, 22."""
import math

REFERENCE_MIXED_UPDATES = 741376
WARMUP_EPOCHS = 0.3

def samples_at_step(step, rows, batch):
    whole, local = divmod(step, math.ceil(rows / batch))
    return whole * rows + min(local * batch, rows)

def endpoint_plan(rows, batch):
    per_epoch = math.ceil(rows / batch)
    ends = {f'epoch_{e:02d}': e * per_epoch for e in (3, 6, 12, 22)}
    return {name: dict(start_step=math.ceil(9 * end / 10), end_step=end,
                      end_examples=samples_at_step(end, rows, batch),
                      effective_epochs=samples_at_step(end, rows, batch) / rows)
            for name, end in ends.items()}

def factor(samples, rows, batch, endpoint=None):
    if not endpoint:
        return min(1., max(0., samples / (WARMUP_EPOCHS * rows)))
    plan = endpoint_plan(rows, batch)[endpoint]
    start = samples_at_step(plan['start_step'], rows, batch)
    progress = min(1., max(0., (samples - start) / (plan['end_examples'] - start)))
    return .5 * (1. + math.cos(math.pi * progress))
