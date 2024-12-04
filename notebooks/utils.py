import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import check_is_fitted

FOLDER_DATETIME_FORMAT = "%Y-%m-%d_%H_%M_%S"
UTC_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class CumulativeIncidencePipeline(Pipeline):
    def predict_cumulative_incidence(self, X, times=None):
        Xt = X
        for _, _, transformer in self._iter(with_final=False):
            Xt = transformer.transform(Xt)
        return self.steps[-1][1].predict_cumulative_incidence(Xt, times)
    
    @property
    def time_grid(self):
        model = self.steps[-1][1]
        check_is_fitted(model, "time_grid_")
        return model.time_grid_


def make_recarray(y):
    # This is an annoying trick to make scikit-survival happy.
    event = y["event"].values
    duration = y["duration"].values
    return np.array(
        [(event[i], duration[i]) for i in range(y.shape[0])],
        dtype=[("e", bool), ("t", float)],
    )


def get_n_events(event):
    return len(set(event.unique()) - {0})
