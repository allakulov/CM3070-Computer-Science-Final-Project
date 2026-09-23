"""Small shared helpers for explicit lot labels. Never infer a lot from a name."""
import re


def lot_key(value):
    """Normalise one label; preserve ambiguous labels instead of taking a digit."""
    text = ' '.join(str(value or '').strip().lower().split())
    if not text or text == 'null':
        return 'all'
    numbers = re.findall(r"\d+", text)
    # A number in prose such as 'iepirkuma 2. daļā' is a lot identifier.
    # Preserve alphanumeric identifiers such as A2 and B2 rather than merging them.
    standalone = re.findall(r"(?<!\w)\d+(?!\w)", text)
    lot_word = re.search(r"\b(?:daļ\w*|lots?)\b", text)
    if len(numbers) == len(standalone) == 1 and (lot_word or text.rstrip(".").isdigit()):
        return str(int(numbers[0]))
    return text


def group_criteria(items):
    """Keep unlabelled items separate; a shared flag never erases explicit lots."""
    groups = {}
    for item in items:
        groups.setdefault(lot_key(item.get('lot')), []).append(item)
    return groups


def scope_error(items, same_for_all_lots=False, lot_ids=()):
    """Reject contradictory scopes and unexpanded numeric lists or ranges."""
    groups = group_criteria(items)
    labels = list(groups) + [lot_key(value) for value in lot_ids]
    for label in labels:
        if len(re.findall(r'\d+', label)) > 1:
            return f'Expand the lot list or range {label!r} into individual lot labels.'
    if same_for_all_lots and any(key != 'all' for key in groups):
        return 'Shared criteria must have null lot labels; do not combine shared and lot-specific sets.'
    if 'all' in groups and len(groups) > 1:
        return 'Unlabelled and lot-specific criteria are mixed; assign their scope from the source.'
    ids = {lot_key(value) for value in lot_ids} - {'all'}
    if not same_for_all_lots and 'all' in groups and len(ids) > 1:
        return 'Several lots are identified but the criteria have no lot labels.'
    return None
