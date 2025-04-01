import numpy as np
from scipy import stats

from alien.sample_generation import (
    Filter,
    RandomSampleGenerator,
    SetSampleGenerator,
    TransformedSampleGenerator,
    UniformSampleGenerator,
)

SEED = 0
N = 1000


class TestRandomSampleGenerator:
    alpha = 1e-3

    def test_normal(self):
        generator = RandomSampleGenerator(distribution="normal", random_seed=SEED)
        samples = generator.generate_samples(N=N)
        _, p_val = stats.normaltest(samples)
        assert (
            p_val > TestRandomSampleGenerator.alpha
        ), f"Reject null-hypothesis with p-value {p_val}"


class TestUniformSampleGenerator:
    alpha = 1e-3

    def test_uniform(self):
        generator = UniformSampleGenerator(random_seed=SEED)
        samples = generator.generate_samples(N=N)
        _, p_val = stats.kstest(samples, stats.uniform().cdf)
        assert (
            p_val > TestUniformSampleGenerator.alpha
        ), f"Reject null-hypothesis with p-value {p_val}"


class TestSetSampleGenerator:
    np.random.seed(SEED)
    n_samples = 64
    n_features = 16
    X = np.random.normal(size=(n_samples, n_features))
    y = np.random.normal(size=n_samples)
    data = {"X": X, "y": y}

    def test_sample(self):
        generator = SetSampleGenerator(TestSetSampleGenerator.data, random_seed=SEED)
        samples = generator.generate_samples()
        assert samples[:, "X"].shape == TestSetSampleGenerator.X.shape


class TestTransformedSampleGenerator:
    @staticmethod
    def f(x):
        return x + 3

    def test_transformation(self):
        generator_base = RandomSampleGenerator(distribution="normal", random_seed=SEED)
        generator = TransformedSampleGenerator(
            generator_base, function=TestTransformedSampleGenerator.f
        )
        samples_base = generator_base.generate_samples(N=N)
        samples = generator.generate_samples(N=N, verbose=False)
        assert np.isclose(np.mean(samples_base), np.mean(samples - 3), atol=5e-2)


class TestFilterGenerator:
    @staticmethod
    def f(x):
        return x + 3

    def test_filter(self):
        generator_base = RandomSampleGenerator(distribution="normal", random_seed=SEED)
        generator = Filter(generator_base, TestFilterGenerator.f, threshold=3)
        samples = generator.generate_samples(N=N)
        assert (samples > 0).all()
