from detectors.base_detector import ConstraintBasedDetector


class GrowShrink(ConstraintBasedDetector):
    """Grow-Shrink (Margaritis & Thrun, NeurIPS 1999).

    Grow: sweep the columns in order and add every one dependent on the target given the current
    blanket, until a sweep adds nothing. Shrink: remove the columns independent given the others.
    Unlike IAMB the order of the additions is the column order, not the strength of the
    association, which lets more false positives in before the shrink.
    """

    base_name = "grow_shrink"

    def find_blanket(self, test, n_columns):
        blanket, changed = [], True
        while changed:
            changed = False
            for j in range(n_columns):
                if j not in blanket and test.pvalue(j, blanket) <= self.alpha:
                    blanket.append(j)
                    changed = True
        return self.shrink(test, blanket)
