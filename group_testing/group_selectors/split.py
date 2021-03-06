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
"""Defines several split group selectors."""

from absl import logging
import gin
from group_testing.group_selectors import group_selector
import jax
import jax.numpy as np
import numpy as onp


@gin.configurable
class SplitSelector(group_selector.GroupSelector):
  """Split the patients into sub groups."""

  def __init__(self, split_factor=None):
    super().__init__()
    self.split_factor = split_factor

  def get_groups(self, rng, state):
    if self.split_factor is None:
      # if no factor is given by default, we use prior infection rate.
      if np.size(state.prior_infection_rate) > 1:
        raise ValueError(
            'Dorfman Splitting cannot be used with individual infection rates.'+
            ' Consider using Informative Dorfman instead.')

      # set group size to value defined by Dorfman testing
      group_size = 1 + np.ceil(1 / np.sqrt(np.squeeze(
          state.prior_infection_rate)))
      # adjust to take into account testing limits
      group_size = min(group_size, state.max_group_size)
      split_factor = -(-state.num_patients // group_size)
    else:
      # ensure the split factor does not produce groups that are too large
      min_splits = -(-state.num_patients // state.max_group_size)
      split_factor = np.maximum(self.split_factor, min_splits)

    indices = onp.array_split(np.arange(state.num_patients), split_factor)
    new_groups = onp.zeros((len(indices), state.num_patients))
    for i in range(len(indices)):
      new_groups[i, indices[i]] = True
    return np.array(new_groups, dtype=bool)

  def __call__(self, rng, state):
    new_groups = self.get_groups(rng, state)
    state.add_groups_to_test(
        new_groups, results_need_clearing=True)
    return state


@gin.configurable
class SplitPositive(group_selector.GroupSelector):
  """First looks for previous groups that were tested positive and not cleared.

  Select and split them using split_factor.
  """

  def __init__(self, split_factor=2):
    super().__init__()
    self.split_factor = split_factor

  def _split_groups(self, groups):
    """Splits the groups."""
    # if split_factor is None, we do exhaustive split,
    # i.e. we test everyone as in Dorfman groups
    use_split_factor = self.split_factor
    # make sure this is a matrix
    groups = onp.atleast_2d(groups)
    n_groups, n_patients = groups.shape

    # we form new groups one by one now. initialize matrix first
    new_groups = onp.empty((0, n_patients), dtype=bool)
    for i in range(n_groups):
      group_i = groups[i, :]
      # test if there is one individual to test
      if np.sum(group_i) > 1:
        indices, = np.where(group_i)
        if self.split_factor is None:
          use_split_factor = np.size(indices)
        indices = onp.array_split(indices, use_split_factor)
        newg = onp.zeros((len(indices), n_patients))
        for j in range(len(indices)):
          newg[j, indices[j]] = 1
      new_groups = onp.concatenate((new_groups, newg), axis=0)
    return np.array(new_groups, dtype=bool)

  def get_groups(self, rng, state):
    if np.size(state.past_groups) > 0 and np.size(state.to_clear_positives) > 0:
      to_split = state.past_groups[state.to_clear_positives, :]
      # we can only split groups that have more than 1 individual
      to_split = to_split[np.sum(to_split, axis=-1) > 1, :]
      if np.size(to_split) > 0:
        if np.ndim(to_split) == 1:
          to_split = onp.expand_dims(to_split, axis=0)
        # each group indexed by indices will be split in split_factor terms
        return self._split_groups(to_split)
      else:
        logging.info('only singletons')

  def __call__(self, rng, state):
    new_groups = self.get_groups(rng, state)

    if new_groups is not None:
      state.add_groups_to_test(new_groups,
                               results_need_clearing=True)
      state.update_to_clear_positives()

    else:
      state.all_cleared = True
    return state


@gin.configurable
class TwoDDorfmanPostSelector(group_selector.GroupSelector):
  """Selects groups following tests described in two_d_dorfman from origamy.py.

  This implements the selector used in the second stage of
  https://www.fda.gov/media/141951/download

  This selector looks at 20 groups extracted from a 8 x 12 assay matrix,
  (8 rows, 12 columns) and takes all individual samples located at
  intersections between these groups to add them back, as individual tests,
  in the tests that need to be tested. If no rows test positive but one or more
  columns test positive (or vice versa), everyone is tested.
  """

  NEEDS_POSTERIOR = False

  def __call__(self, rng, state):
    """Produces new groups from 1st wave of results & adds them to stack."""
    num_rows = 8
    num_cols = 12
    # check this is the first time selector is called (or not used)
    if state.past_groups.shape[0] == num_rows + num_cols:
      # sum groups that have returned positive
      returned_positive = np.sum(
          state.past_groups[state.past_test_results, :], axis=0)
      total_positive_first_block = np.sum(state.past_test_results[0:num_rows],
                                          axis=0)
      total_positive_second_block = np.sum(
          state.past_test_results[num_rows:num_rows+num_cols], axis=0)
      # check if incoherence in row/columns
      if np.logical_xor(total_positive_first_block > 0,
                        total_positive_second_block > 0):
        # test everyone that was ever included in a positive group
        reflex_to_test, = np.where(returned_positive)
      else:
        # test individually those that returned positive at least twice.
        reflex_to_test, = np.where(returned_positive > 1)
      new_groups = jax.nn.one_hot(reflex_to_test, state.num_patients).astype(bool)
      state.add_groups_to_test(new_groups)
      logging.warning('Added %i groups to test', new_groups.shape[0])
      logging.debug(new_groups.astype(np.int32))
    else:
      state.all_cleared = True
    return state
