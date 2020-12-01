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
"""Implements our Informative Dorfman group selector."""

import gin
import jax
import jax.numpy as np
import numpy as onp


from group_testing import utils
from group_testing.group_selectors import group_selector


@gin.configurable
class InformativeDorfman(group_selector.GroupSelector):
  """Implements PSOD from (McMahan & al, 2012) paper, with modifications.

  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3197971/ S.3.3 PSOD
  http://www.biometrics.tibs.org/datasets/100838PR2_supp_materials.pdf.

  Other reference:
  R implementation
  https://www.rdocumentation.org/packages/binGroup/versions/2.2-1/source
  in Dorfman.Functions.R, line 430
  pool.specific.dorf<-function(p,max.p,se,sp)

  Added a cut_off_low / cut_off_high parameter to stop testing patients
  whose marginal is already more or less decided, as well as random sampling
  to account for testing capacities.
  """

  NEEDS_POSTERIOR = False

  def __init__(self,
               cut_off_high: float = 1,
               cut_off_low: float = 0,
               modified: bool = False):
    """Creates an Informative Dorfman selector.

    Args:
      cut_off_high : float, patients with a marginal value above that threshold
        are not considered.
      cut_off_low : float, patients with a marginal value below that threshold
        are not considered.
      modified : bool that indicates whether the ID selector (from original
        paper) is modified to only output a predetermined number of tests, in
        which case the idea is to randomly sample a subset of them, or whether
        we proceed in standard fashion, as is the case with Dorfman strategies,
        by adding all tests that are needed regardless of the capacity of the
        testing device in the next testing cycle.
    """

    super().__init__()
    self.modified = modified
    self.cut_off_high = cut_off_high
    self.cut_off_low = cut_off_low

  def __call__(self, rng, state):
    """Produces new groups and adds them to state's stack."""
    p_weights, particles = state.particle_weights, state.particles
    marginal = onp.array(np.sum(p_weights[:, np.newaxis] * particles, axis=0))
    marginal = onp.squeeze(marginal)
    not_cut_ids, = onp.where(np.logical_and(
        marginal < self.cut_off_high, marginal > self.cut_off_low))
    marginal = marginal[not_cut_ids]
    sorted_ids = onp.argsort(marginal)
    sorted_marginal = onp.array(marginal[sorted_ids])
    n_p = 0
    n_r = marginal.size
    if n_r == 0:  # no one left to test in between thresholds
      state.all_cleared = True
      return state

    all_new_groups = np.empty((0, state.num_patients), dtype=bool)
    while n_p < marginal.size:
      index_max = onp.amin((n_r, state.max_group_size))
      group_sizes = onp.arange(1, index_max + 1)
      cum_prod_prob = onp.cumprod(1 - sorted_marginal[n_p:(n_p + index_max)])
      # formula below is only valid for group_size > 1,
      # corrected below for a group of size 1.
      sensitivity = onp.array(
          utils.select_from_sizes(state.prior_sensitivity, group_sizes))
      specificity = onp.array(
          utils.select_from_sizes(state.prior_specificity, group_sizes))

      exp_div_size = (
          1 + group_sizes *
          (sensitivity + (1 - sensitivity - specificity) * cum_prod_prob)
          ) / group_sizes
      exp_div_size[0] = 1  # adjusted cost for one patient is one.
      opt_size_group = onp.argmin(exp_div_size) + 1
      new_group = onp.zeros((1, state.num_patients))
      new_group[0, not_cut_ids[sorted_ids[n_p:n_p + opt_size_group]]] = True
      all_new_groups = np.concatenate((all_new_groups, new_group), axis=0)
      n_p = n_p + opt_size_group
      n_r = n_r - opt_size_group
    # sample randomly extra_tests_needed groups in modified case, all in
    # regular ID.
    # Because ID is a Dorfman type approach, it might be followed
    # by exhaustive splitting, which requires to keep track of groups
    # that tested positives to retest them.
    all_new_groups = jax.random.permutation(rng, all_new_groups)
    if self.modified:
      # in the case where we use modified ID, we only subsample a few groups.
      # one needs to take care of requesting to keep track of positives.
      new_groups = all_new_groups[0:state.extra_tests_needed].astype(bool)
      state.add_groups_to_test(new_groups,
                               results_need_clearing=True)
    else:
      # with regular ID we add all groups at once.
      state.add_groups_to_test(all_new_groups.astype(bool),
                               results_need_clearing=True)
    return state
