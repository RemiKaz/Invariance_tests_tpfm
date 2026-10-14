from detectors.base_detector import ConstraintBasedDetector


class IAMB(ConstraintBasedDetector):
    """Incremental Association Markov Blanket (Tsamardinos, Aliferis & Statnikov, FLAIRS 2003).

    Forward: add, one at a time, the column most dependent on the target given the current
    blanket (the smallest p-value), as long as it is significant. Backward: remove the columns
    that became independent given the others. Sound and complete under faithfulness and with
    correct tests.
    """

    base_name = "iamb"

    def find_blanket(self, test, n_columns):
        blanket = []
        while len(blanket) < n_columns:
            pvalues = {j: test.pvalue(j, blanket) for j in range(n_columns) if j not in blanket}
            best = min(pvalues, key=pvalues.get)
            if pvalues[best] > self.alpha:
                break
            blanket.append(best)
        return self.shrink(test, blanket)
