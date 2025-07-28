import torch
import gpytorch
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)  # type: ignore


class TanimotoGP:
    def __init__(self):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = None
        self.train_x = None
        self.train_y = None

    def fit(self, X, y):
        self.train_x = torch.tensor(X, dtype=torch.float)
        self.train_y = torch.tensor(y, dtype=torch.float)

        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(50):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -1 * mll(output, self.train_y)  # type: ignore
            loss.backward()  # type: ignore
            optimizer.step()

    def predict(self, X):
        self.model.eval()  # type: ignore
        self.likelihood.eval()

        test_x = torch.tensor(X, dtype=torch.float)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(test_x))  # type: ignore
            return preds.mean.numpy(), preds.variance.numpy()
