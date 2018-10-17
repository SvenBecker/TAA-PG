from tensorforce.core.baselines import Baseline


# In progress
class TAABaseline(Baseline):
    """
    Buy and hold baseline
    """
    def __init__(self, scope='bah-baseline', summary_labels=None):
        super(TAABaseline, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_predict(self, **kwargs):
        pass





