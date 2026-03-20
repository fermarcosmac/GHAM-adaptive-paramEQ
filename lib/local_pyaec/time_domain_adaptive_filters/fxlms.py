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

def fxlms(x, d, h_hat, N = 4, mu = 0.1):
  x = np.asarray(x, dtype=float)
  d = np.asarray(d, dtype=float)
  h_hat = np.asarray(h_hat, dtype=float)

  N = int(N)
  if N <= 0:
    raise ValueError("N must be > 0")
  if h_hat.size == 0:
    raise ValueError("h_hat must not be empty")

  nIters = min(len(x), len(d))
  if nIters <= 0:
    return np.zeros(0)

  # Regressor for control filter output y[n] = w^T u[n].
  u = np.zeros(N)
  # Regressor built from filtered-x signal x_f[n] = (h_hat * x)[n].
  u_f = np.zeros(N)
  # State for online FIR filtering by h_hat.
  x_state = np.zeros(h_hat.size)

  w = np.zeros(N)
  e = np.zeros(nIters)

  for n in range(nIters):
    # Raw reference vector for control output.
    u[1:] = u[:-1]
    u[0] = x[n]

    # Compute filtered-x sample with estimated secondary path.
    x_state[1:] = x_state[:-1]
    x_state[0] = x[n]
    x_f_n = np.dot(h_hat, x_state)

    # Filtered reference vector used in gradient update.
    u_f[1:] = u_f[:-1]
    u_f[0] = x_f_n

    y_n = np.dot(w, u)
    e_n = d[n] - y_n
    w = w + mu * e_n * u_f
    e[n] = e_n

  return e, w # Return error progress and final control filter coefficients.