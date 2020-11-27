# coding=utf-8
# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Dorfman Deterministic Decoder."""

import gin
from group_testing.samplers import sampler
import jax.numpy as np


@gin.configurable
class DorfmanDecoder(sampler.Sampler):
  """Dorfman deterministic decoding: only positive individual tests recorded."""

  NAME = "DORF"

  def produce_sample(self, rng, state):
    """Produces only "one" fractional particle state: a marginal.

    Args:
      rng : random PRNG key. Not used.
      state : state object containing all relevant information to produce sample

    Returns:
      a measure of the quality of convergence, here gap_between_consecutives
      also updates particle_weights and particles members.
    """
    self.particle_weights = np.array([1])
    marginal = np.zeros((state.num_patients,))
    tests = state.past_test_results
    groups = state.past_groups

    # bool index of groups that only test individuals and that returned positive
    ind_tests = (np.sum(groups, axis=1) == 1) * tests

    # positives is the sum of those groups, all added if at least 1 was found.
    if np.sum(ind_tests):
      marginal += np.sum(groups[ind_tests, :], axis=0)
    # this is the single particle that we refresh in state.
    self.particles = np.expand_dims(marginal, axis=0)


