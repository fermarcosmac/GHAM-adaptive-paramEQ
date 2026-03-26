# Copyright 2020 ewan xu<ewan_xu@outlook.com>
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
# =============================================================================

""" Least Mean Squares Filter """

import numpy as np

def fxlms(x, d, h_hat, N = 4, mu = 0.1, w_init:np.ndarray = None, u_state = None, u_f_state = None, x_state = None):
  x = np.asarray(x, dtype=float)
  d = np.asarray(d, dtype=float)
  h_hat = np.asarray(h_hat, dtype=float)
  if w_init is None:
    w = np.zeros(N)
  else: # if provided, don't initialize controller filter in each call
    w = np.asarray(w_init, dtype=float)

  N = int(N)
  if N <= 0:
    raise ValueError("N must be > 0")
  if h_hat.size == 0:
    raise ValueError("h_hat must not be empty")

  nIters = min(len(x), len(d))
  if nIters <= 0:
    return np.zeros(0)

  # Regressor for control filter output y[n] = w^T u[n].
  u = np.zeros(N) if u_state is None else u_state
  # Regressor built from filtered-x signal x_f[n] = (h_hat * x)[n].
  u_f = np.zeros(N) if u_f_state is None else u_f_state
  # State for online FIR filtering by h_hat.
  x_state = np.zeros(h_hat.size) if x_state is None else x_state

  e = np.zeros(nIters)

  for n in range(nIters):
    # u and x_state are both buffers for input, but with different lengths (possibly different memoried of direct path and controller filter)
    # Raw reference vector for control output.
    u[1:] = u[:-1]
    u[0] = x[n]

    # Compute filtered-x sample with estimated secondary path.
    x_state[1:] = x_state[:-1]
    x_state[0] = x[n]

    # Filtered reference vector used in gradient update.
    u_f[1:] = u_f[:-1]
    u_f[0] = np.dot(h_hat, x_state)

    y_n = np.dot(w, u)
    e_n = d[n] - y_n
    w = w + mu * e_n * u_f
    e[n] = e_n

  return e, w, u_state, u_f_state, x_state # Return error progress, final control filter coefficients and state variables.