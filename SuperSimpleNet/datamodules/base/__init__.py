from enum import Enum


class Supervision(str, Enum):
    UNSUPERVISED = "unsupervised"
    WEAKLY_SUPERVISED = "weakly_supervised"
    MIXED_SUPERVISION = "mixed_supervision"
    FULLY_SUPERVISED = "fully_supervised"


def segmented2segmented(supervision, is_segmented):
    """Correct the is_segmented value according to supervision.

    Enables compatibility with splits of form (sample, is_segmented)
    where is_segmented also serves to say if sample is labeled (KSDD2, base Sensum splits).
    Args:
        supervision (Supervision): Supervision enum
        is_segmented (bool): is_segmented from split file

    Returns:
        bool: corrected is_segmented value
    """
    # Weakly supervised is always not segmented, others respect the given value.
    if supervision == Supervision.WEAKLY_SUPERVISED:
        return False
    else:
        return is_segmented
