import torch
import numpy as np


def initial_values_sampler_uniform(batch_size, space_dim, space_bounds):
    # samples points uniformly in hypercube [a, b ] ^dim
    a, b = space_bounds
    return (b - a) * torch.rand([batch_size, space_dim]) + a


def initial_values_sampler_gaussian(batch_size, space_dim):
    # samples standard normally distributed points in R^d
    return torch.randn([batch_size, space_dim])


def initial_values_sampler_rectangular(batch_size, space_dim, space_bounds):
    # samples points uniformly in hyper-cuboid [a_1, b_1] x ... x [a_d, b_d]
    return space_bounds * np.random.uniform(0., 1., (batch_size, space_dim))


class RectangleValueSampler:
    # class implementing uniform sampling from hyper-cuboid.
    def __init__(self, space_dim, space_bounds):
        self.space_dim = space_dim
        assert len(space_bounds) == space_dim
        self.space_bounds = space_bounds

    def sample(self, batch_size):
        values = initial_values_sampler_rectangular(batch_size, self.space_dim, self.space_bounds)
        return values


class UniformValueSampler:
    # class implementing uniform sampling from hyper-cube.
    def __init__(self, space_dim, space_bounds, dev):
        self.space_dim = space_dim

        assert len(space_bounds) == 2
        self.space_bounds = space_bounds
        self.dev = dev

    def sample(self, batch_size):
        values = initial_values_sampler_uniform(batch_size, self.space_dim, self.space_bounds).to(self.dev)
        return values


class NormalValueSampler:
    # class implementing sampling of standard normal values.
    def __init__(self, space_dim, dev):
        self.space_dim = space_dim
        self.dev = dev

    def sample(self, batch_size):
        values = initial_values_sampler_gaussian(batch_size, self.space_dim).to(self.dev)
        return values


class DataSampler:
    # generates data from dataset loaded via data_loader.
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.generator = iter(data_loader)

    def sample(self, batch_size):
        try:
            x = next(self.generator)
        except StopIteration:
            self.generator = iter(self.data_loader) # if the generator is empty: Create new generator.
            x = next(self.generator)
        return x


class DrawBatch:
    # Draws a batch from given data tensor by randomly selecting indices.
    def __init__(self, data):
        self.data = data
        self.len = data.shape[0]

    def sample(self, bs):
        indices = torch.randint(0, self.len, (bs, ))
        return self.data[indices, :]