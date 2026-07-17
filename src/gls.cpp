/*
 * gls.cpp
 *
 *  Created on: 16 Jul 2026
 *      Author: radu
 *
 * Copyright (c) 2015, International Business Machines Corporation
 * and University of California Irvine. All rights reserved.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/// \file gls.cpp
/// \brief Guided Local Search (GLS+) for MAP inference
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#include "gls.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace merlin {

// Weighted penalty delta of flipping var to val, summed over the factors var
// occurs in: -m_penalty_mult * Σ (penalty[new entry] - penalty[old entry]).
// Flipping toward a lower-penalty entry (negative raw delta) raises the score.
double gls::penalty_delta(vindex var, size_t val) const {
	if (val == m_value[var]) return 0.0;
	size_t cur = m_value[var];
	double dp = 0.0;
	for (size_t j = 0; j < m_occ[var].size(); ++j) {
		findex f = m_occ[var][j].first;
		// For MMAP, only pure-MAP factors carry penalties (mixed/SUM factors are
		// handled by the WMB conditioned-SUM estimate, not penalized).
		if (m_task == Task::MMAP && !m_is_map_factor[f]) continue;
		size_t pos = m_occ[var][j].second;
		size_t base = m_index[f];
		long shift = (long) m_stride[f][pos] * ((long) val - (long) cur);
		dp += m_penalty[f][base + shift] - m_penalty[f][base];
	}
	return -m_penalty_mult * dp;
}

// MMAP move score: the pure MMAP objective delta plus the guided-search penalty
// delta. (For MMAP, m_index reflects the current MAP assignment because eval_mmap
// / the driver keep it via the factor tables; penalties are read on demand.)
double gls::score_mmap(vindex var, size_t val) {
	return sls::score_mmap(var, val) + penalty_delta(var, val);
}

// Utility of factor f's current entry, driving which entries get penalized:
// high for high-cost (very negative log) and lightly-penalized entries, so
// penalizing them most reshapes the surface. Cost = -logCost = -m_log_factor.
double gls::utility(findex f) const {
	double log_val = m_log_factor[f][m_index[f]];
	double pen = m_penalty[f][m_index[f]];
	return (-log_val) / (1.0 + pen); // -log_val >= 0 for probabilities <= 1
}

// At a local minimum, increment the penalty of the currently-instantiated entries
// with maximal utility, and incrementally update the affected variables' penalty
// scores and good-variable status. Mirrors SLS4MPE's increasePenalties +
// incCurrPenalty.
void gls::increase_penalties() {
	size_t nf = m_gmo.num_factors();
	bool mmap = (m_task == Task::MMAP);

	// Determine the maximal utility over the current factor entries. For MMAP only
	// pure-MAP factors are eligible (their current entry is well-defined).
	double max_util = -std::numeric_limits<double>::infinity();
	for (findex f = 0; f < nf; ++f) {
		if (mmap && !m_is_map_factor[f]) continue;
		max_util = std::max(max_util, utility(f));
	}
	if (max_util == -std::numeric_limits<double>::infinity()) return; // nothing to penalize

	// Penalize every current entry achieving (approximately) the max utility.
	for (findex f = 0; f < nf; ++f) {
		if (mmap && !m_is_map_factor[f]) continue;
		if (std::fabs(utility(f) - max_util) >= 1e-9) continue;
		size_t idx = m_index[f];
		m_penalty[f][idx] += m_penalty_increment;

		// The current entry got more penalized, so for every scope variable u,
		// flipping u away from its current value now moves to a relatively-lower-
		// penalty entry. For MAP the cached m_pen_score is updated incrementally;
		// for MMAP there is no penalty cache (scores are read on demand via
		// penalty_delta), so only the good-variable set is refreshed.
		for (size_t q = 0; q < m_scope[f].size(); ++q) {
			vindex u = m_scope[f][q];
			if (!mmap) {
				size_t cur = m_value[u];
				for (size_t w = 0; w < m_dom[u]; ++w) {
					if (w == cur) continue;
					m_pen_score[u][w] -= m_penalty_increment;
				}
			}
			// A pure-MAP factor's scope is all MAP vars, so u is a MAP var here.
			update_good_var(u);
		}
	}
}

// Scale all penalties toward zero (smoothing) and rescale the cached penalty
// scores by the same factor so they stay consistent without a full recompute.
void gls::scale_penalties(double factor) {
	size_t nf = m_gmo.num_factors();
	for (findex f = 0; f < nf; ++f)
		for (size_t i = 0; i < m_penalty[f].size(); ++i)
			m_penalty[f][i] *= factor;
	// For MAP, rescale the cached penalty scores too; MMAP has no such cache
	// (penalty deltas are read on demand from m_penalty).
	if (m_task != Task::MMAP) {
		size_t n = m_gmo.nvar();
		for (size_t v = 0; v < n; ++v)
			for (size_t val = 0; val < m_dom[v]; ++val)
				m_pen_score[v][val] *= factor;
	}
}

void gls::init() {
	std::cout << "[GLS+] + inference task   : "
			<< (m_task == Task::MMAP ? "MMAP" : "MAP") << std::endl;
	std::cout << "[GLS+] + time limit       : " << m_time_limit << std::endl;
	std::cout << "[GLS+] + max flips        : " << m_max_flips << std::endl;
	std::cout << "[GLS+] + penalty inc/mult : " << m_penalty_increment << " / "
			<< m_penalty_mult << std::endl;
	std::cout << "[GLS+] + smooth/interval  : " << m_smooth << " / " << m_interval
			<< std::endl;
	std::cout << "[GLS+] + seed             : " << m_seed << std::endl;

	LOG_ZERO = -1.0e6;
	rand_seed(m_seed);
	build_structures();

	// Allocate the penalty tables (all zero). Their presence switches the shared
	// engine (recompute_scores / flip_to) into tracking m_pen_score.
	size_t nf = m_gmo.num_factors();
	m_penalty.assign(nf, std::vector<double>());
	for (findex f = 0; f < nf; ++f)
		m_penalty[f].assign(m_gmo.get_factor(f).numel(), 0.0);

	size_t n = m_gmo.nvar();
	m_value.assign(n, 0);
	m_best_config.assign(n, 0);
	m_logz = -std::numeric_limits<double>::infinity();
	m_found_solution = false;
	m_num_flips = 0;
	m_num_flips_total = 0;
	m_num_restarts = 0;
	m_num_evals = 0;
}

// The GLS+ main loop: greedy best flip by the augmented (log-prob + penalty)
// score; at a local minimum, increase penalties (periodically smoothing them).
// No random restarts. Runs until the time/flip budget is exhausted.
void gls::run() {
	m_start_time = timeSystem();
	init();

	// Initial random assignment + score seed.
	random_restart();
	m_num_restarts = 0;
	update_best();

	long lm_counter = 0;
	long recompute_interval = 100000;

	while (ls_continue()) {
		if (m_num_flips > 0 && (m_num_flips % recompute_interval) == 0)
			recompute_scores();

		vindex var = (vindex) -1;
		size_t val = 0;
		if (best_new_inst(var, val)) {
			flip_to(var, val);
		} else {
			// Local minimum: reshape the surface via penalties.
			++lm_counter;
			if (m_interval > 0 && (lm_counter % m_interval) == 0)
				scale_penalties(m_smooth);
			increase_penalties();
			++m_num_flips; // count so a pure flip budget still terminates
		}
		update_best();
	}

	m_logz = m_found_solution ? m_logz : -std::numeric_limits<double>::infinity();

	std::cout << "[GLS+] Finished searching in "
			<< (timeSystem() - m_start_time) << " seconds" << std::endl;
	std::cout << "[GLS+] + flips            : " << m_num_flips_total << std::endl;
	std::cout << "[GLS+] + local minima     : " << lm_counter << std::endl;
	std::cout << "[GLS+] Best value found   : " << m_logz
			<< " (" << std::exp(m_logz) << ")" << std::endl;
}

} // namespace merlin
