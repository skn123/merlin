/*
 * sls.cpp
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

/// \file sls.cpp
/// \brief Stochastic Local Search (SLS / G+StS) for MAP inference
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#include "sls.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace merlin {

// Tolerance below which a score is not considered a strict improvement. Matches
// the ground-truth EPS used to avoid endless loops from floating-point noise.
static const double SLS_EPS = 1e-6;

// Build the per-factor log tables, index strides, per-variable occurrence lists,
// and Markov blankets. Called once from init(). Values are natural logs; a zero
// probability is clamped to LOG_ZERO (a large finite negative) so the objective
// stays finite and no assignment can produce -inf/NaN scores.
void sls::build_structures() {
	size_t n = m_gmo.nvar();
	size_t nf = m_gmo.num_factors();

	m_dom.assign(n, 0);
	for (size_t v = 0; v < n; ++v) m_dom[v] = m_gmo.var(v).states();

	m_log_factor.assign(nf, std::vector<double>());
	m_scope.assign(nf, std::vector<size_t>());
	m_stride.assign(nf, std::vector<size_t>());
	m_index.assign(nf, 0);
	m_occ.assign(n, std::vector<std::pair<findex, size_t> >());

	for (findex f = 0; f < nf; ++f) {
		const factor& fac = m_gmo.get_factor(f);
		const variable_set& sc = fac.vars();
		size_t sz = fac.numel();
		// Log table (clamped for zero-probability entries).
		m_log_factor[f].resize(sz);
		for (size_t i = 0; i < sz; ++i) {
			double v = fac[i];
			m_log_factor[f][i] = (v <= 0.0) ? LOG_ZERO : std::log(v);
		}
		// Scope (variable_set order) + index strides (product of preceding domains).
		size_t stride = 1, p = 0;
		for (variable_set::const_iterator vi = sc.begin(); vi != sc.end(); ++vi, ++p) {
			m_scope[f].push_back(vi->label());
			m_stride[f].push_back(stride);
			stride *= vi->states();
		}
		// Occurrence list: variable -> (factor, local position).
		for (size_t q = 0; q < m_scope[f].size(); ++q)
			m_occ[m_scope[f][q]].push_back(std::make_pair(f, q));
	}

	// Markov blanket of each variable (variables sharing a factor with it,
	// including itself), used to know whose scores an incremental flip affects.
	m_mb.assign(n, std::vector<vindex>());
	std::vector<char> seen(n, 0);
	for (size_t v = 0; v < n; ++v) {
		for (size_t j = 0; j < m_occ[v].size(); ++j) {
			findex f = m_occ[v][j].first;
			for (size_t q = 0; q < m_scope[f].size(); ++q) {
				vindex u = m_scope[f][q];
				if (!seen[u]) { seen[u] = 1; m_mb[v].push_back(u); }
			}
		}
		for (size_t j = 0; j < m_mb[v].size(); ++j) seen[m_mb[v][j]] = 0; // reset
	}

	// Classify MAP (query) vs SUM variables. For MAP the query is all variables;
	// for MMAP only the query variables are MAP and the local search flips those.
	m_is_map.assign(n, 0);
	m_map_vars.clear();
	if (m_task == Task::MMAP) {
		for (size_t i = 0; i < m_query.size(); ++i) m_is_map[m_query[i]] = 1;
	} else {
		for (size_t v = 0; v < n; ++v) m_is_map[v] = 1;
	}
	for (size_t v = 0; v < n; ++v) if (m_is_map[v]) m_map_vars.push_back(v);

	if (m_task == Task::MMAP) build_mmap_structures();
}

// Build the WMB heuristic (constrained order with SUM variables eliminated first,
// so MAP vars sit at the top of the pseudo tree), the pseudo-tree structures, the
// per-variable subtree normalization used to turn get_heuristic into an absolute
// conditioned-SUM bound, the factor-anchoring, and the list of SUM-roots (SUM
// variables whose pseudo-tree parent is a MAP variable or a pseudo-tree root).
// This mirrors aobb::init()'s heuristic setup, reused here to score MMAP
// assignments by their (bounded) probability of evidence over the SUM part.
void sls::build_mmap_structures() {
	size_t n = m_gmo.nvar();
	m_var_types.assign(n, false);
	for (size_t i = 0; i < m_query.size(); ++i) m_var_types[m_query[i]] = true;

	// Constrained elimination order + pseudo tree.
	m_order = m_gmo.order2(graphical_model::OrderMethod::MinFill, m_var_types);
	m_parents = m_gmo.pseudo_tree(m_order);
	m_children.assign(n, std::vector<vindex>());
	std::vector<vindex> roots;
	for (size_t v = 0; v < n; ++v) {
		vindex p = m_parents[v];
		if (p == (vindex) -1) roots.push_back(v);
		else m_children[p].push_back(v);
	}
	m_position.assign(n, 0);
	for (size_t i = 0; i < m_order.size(); ++i) m_position[m_order[i]] = i;

	// Anchor each factor at its smallest-position (deepest) scope variable.
	m_anchored.assign(n, flist());
	for (size_t f = 0; f < m_gmo.num_factors(); ++f) {
		const variable_set& sc = m_gmo.get_factor(f).vars();
		if (sc.nvar() == 0) continue;
		vindex anchor = sc.begin()->label();
		size_t best_pos = m_position[anchor];
		for (variable_set::const_iterator vi = sc.begin(); vi != sc.end(); ++vi) {
			if (m_position[vi->label()] <= best_pos) {
				best_pos = m_position[vi->label()];
				anchor = vi->label();
			}
		}
		m_anchored[anchor] |= f;
	}

	// Build and run the WMB heuristic on the same order/pseudo tree.
	m_heuristic = wmb(m_gmo);
	std::ostringstream oss;
	oss << "iBound=" << m_ibound << ",Order=MinFill,Iter=10,Task=MMAP,Debug=0";
	m_heuristic.set_properties(oss.str());
	m_heuristic.set_var_types(m_var_types);
	m_heuristic.set_query(m_query);
	m_heuristic.set_order(m_order);
	m_heuristic.set_pseudo_tree(m_parents);
	m_heuristic.run();

	// Per-variable subtree normalization (leaves -> root sum of bucket_norm), so
	// -log(get_heuristic(x)) - subtree_norm[x] is an absolute conditioned bound.
	m_subtree_norm.assign(n, 0.0);
	for (size_t oi = 0; oi < m_order.size(); ++oi) {
		vindex x = m_order[oi];
		m_subtree_norm[x] += m_heuristic.bucket_norm(x);
		vindex p = m_parents[x];
		if (p != (vindex) -1) m_subtree_norm[p] += m_subtree_norm[x];
	}

	// SUM-roots: SUM variables whose pseudo-tree parent is a MAP variable (or is a
	// root). Each is the top of a conditioned-SUM subtree evaluated by the WMB
	// estimator given the current MAP assignment.
	m_sum_roots.clear();
	for (size_t v = 0; v < n; ++v) {
		if (m_var_types[v]) continue; // MAP var
		vindex p = m_parents[v];
		if (p == (vindex) -1 || m_var_types[p]) m_sum_roots.push_back(v);
	}

	// Pure-MAP factors: scope is entirely MAP variables, so the factor's value is
	// fully determined by the MAP assignment. Only these are penalized by GLS+ in
	// MMAP (their index is well-defined); mixed/SUM factors are handled by the WMB
	// conditioned-SUM estimate, not by penalties.
	m_is_map_factor.assign(m_gmo.num_factors(), 0);
	for (size_t f = 0; f < m_gmo.num_factors(); ++f) {
		bool all_map = true;
		for (size_t q = 0; q < m_scope[f].size(); ++q)
			if (!m_var_types[m_scope[f][q]]) { all_map = false; break; }
		m_is_map_factor[f] = all_map ? 1 : 0;
	}
}

// WMB conditioned-SUM cost estimate for the subtree rooted at SUM variable x,
// given the MAP ancestors set in asgn (cost space, -log). This is exactly
// aobb::heuristic_or for a SUM OR node: for each value v of x, cost[v] =
// -log(label(x=v)) + (-log(get_heuristic(x)) - subtree_norm[x]); the node's SUM
// estimate is -logsumexp(-cost[v]). Uses the shared LOG_ZERO clamp for dead ends.
double sls::sum_estimate(vindex x, std::vector<size_t>& asgn) {
	const double INF = std::numeric_limits<double>::infinity();
	// The WMB heuristic at the SUM-root x (with x itself UNSET and its MAP
	// ancestors fixed in asgn) bounds the conditioned partition of x's ENTIRE
	// subtree: get_heuristic(x, asgn) is a linear-space upper bound on
	// Σ_{x's subtree} P(...|MAP ancestors). The absolute cost is
	// -log(get_heuristic(x)) - subtree_norm[x] (the subtree-normalization strips
	// the message constants the WMB forward pass folded into logZ). One call --
	// no per-value loop (that would re-sum the x level already inside the bound).
	size_t saved = asgn[x];
	asgn[x] = (size_t) -1; // free x so the bound covers the whole subtree
	double hv = m_heuristic.get_heuristic(x, asgn);
	asgn[x] = saved;
	if (hv <= 0.0) return INF; // zero-probability conditioned subtree
	return -std::log(hv) - m_subtree_norm[x];
}

// Estimate Q(x_M) = log P(x_M, evidence) for the MAP assignment in asgn (MAP vars
// set, SUM vars == -1). Q = exact log of the factors anchored at MAP variables
// plus the WMB conditioned-SUM estimate over each SUM-root subtree. Returned in
// log-probability space (higher is better; the search maximizes it).
double sls::eval_mmap(std::vector<size_t>& asgn) {
	++m_num_evals;
	const double INF = std::numeric_limits<double>::infinity();
	// Exact MAP part: factors anchored at MAP variables, fully instantiated.
	double cost = 0.0;
	for (size_t k = 0; k < m_map_vars.size(); ++k) {
		vindex v = m_map_vars[k];
		const flist& fs = m_anchored[v];
		for (flist::const_iterator it = fs.begin(); it != fs.end(); ++it) {
			const factor& fac = m_gmo.get_factor(*it);
			double fv = fac[sub2ind(fac.vars(), asgn)];
			cost += (fv <= 0.0) ? (-LOG_ZERO) : -std::log(fv);
		}
	}
	// Conditioned-SUM part: WMB estimate over each SUM-root subtree.
	for (size_t k = 0; k < m_sum_roots.size(); ++k) {
		double c = sum_estimate(m_sum_roots[k], asgn);
		if (c == INF) return -(-LOG_ZERO); // dead end -> very low log-prob
		cost += c;
	}
	return -cost; // log-prob = -cost
}

// Score of flipping MAP variable var to val under the MMAP objective:
// Q(with var=val) - Q(current). Positive => improving.
double sls::score_mmap(vindex var, size_t val) {
	if (val == m_value[var]) return 0.0;
	size_t saved = m_value[var];
	m_value[var] = val;
	double q_new = eval_mmap(m_value);
	m_value[var] = saved;
	return q_new - m_log_prob; // m_log_prob holds current Q(x_M)
}

// Full recompute of the current factor indices, per-variable per-value score
// deltas, the good-variable set, and the current log-probability. Used to seed
// the search after (re)initialization and periodically to cancel accumulated
// floating-point drift from the incremental updates.
void sls::recompute_scores() {
	size_t n = m_gmo.nvar();
	size_t nf = m_gmo.num_factors();
	bool gls = !m_penalty.empty();

	if (m_task == Task::MMAP) {
		// MMAP: the objective is not factor-decomposable over the MAP variables,
		// so there is no incremental delta cache. Keep the factor indices current
		// (SUM vars unset count as value 0 in the index; only MAP-factor penalties
		// read them), refresh the objective, and rebuild the good-variable set.
		for (findex f = 0; f < nf; ++f) {
			size_t idx = 0;
			for (size_t q = 0; q < m_scope[f].size(); ++q) {
				size_t vv = m_value[m_scope[f][q]];
				if (vv == (size_t) -1) vv = 0; // unset SUM var
				idx += m_stride[f][q] * vv;
			}
			m_index[f] = idx;
		}
		m_log_prob = eval_mmap(m_value);
		m_good_vars.clear();
		m_is_good.assign(n, 0);
		for (size_t k = 0; k < m_map_vars.size(); ++k)
			update_good_var(m_map_vars[k]);
		return;
	}

	// Factor indices for the current assignment.
	for (findex f = 0; f < nf; ++f) {
		size_t idx = 0;
		for (size_t q = 0; q < m_scope[f].size(); ++q)
			idx += m_stride[f][q] * m_value[m_scope[f][q]];
		m_index[f] = idx;
	}

	// Current log-probability.
	m_log_prob = 0.0;
	for (findex f = 0; f < nf; ++f)
		m_log_prob += m_log_factor[f][m_index[f]];

	// Per-value score deltas for every variable: Σ over its factors of the change
	// in that factor's log-value (and penalty for GLS+) when the variable flips.
	m_log_score.assign(n, std::vector<double>());
	if (gls) m_pen_score.assign(n, std::vector<double>());
	m_good_vars.clear();
	m_is_good.assign(n, 0);
	for (size_t v = 0; v < n; ++v) {
		m_log_score[v].assign(m_dom[v], 0.0);
		if (gls) m_pen_score[v].assign(m_dom[v], 0.0);
		size_t cur = m_value[v];
		for (size_t val = 0; val < m_dom[v]; ++val) {
			if (val == cur) continue;
			double dl = 0.0, dp = 0.0;
			for (size_t j = 0; j < m_occ[v].size(); ++j) {
				findex f = m_occ[v][j].first;
				size_t pos = m_occ[v][j].second;
				size_t base = m_index[f];
				long shift = (long) m_stride[f][pos] * ((long) val - (long) cur);
				dl += m_log_factor[f][base + shift] - m_log_factor[f][base];
				if (gls)
					dp += m_penalty[f][base + shift] - m_penalty[f][base];
			}
			m_log_score[v][val] = dl;
			if (gls) m_pen_score[v][val] = dp;
		}
		update_good_var(v);
	}
}

// Recompute whether variable v has a strictly improving flip and update the set.
void sls::update_good_var(vindex v) {
	double best = -std::numeric_limits<double>::infinity();
	for (size_t val = 0; val < m_dom[v]; ++val) {
		if (val == m_value[v]) continue;
		double s = (m_task == Task::MMAP) ? score_mmap(v, val) : score(v, val);
		best = std::max(best, s);
	}
	bool good = (best > SLS_EPS);
	if (good && !m_is_good[v]) {
		m_is_good[v] = 1;
		m_good_vars.push_back(v);
	} else if (!good && m_is_good[v]) {
		m_is_good[v] = 0;
		// Remove v from m_good_vars (swap-with-last).
		for (size_t k = 0; k < m_good_vars.size(); ++k) {
			if (m_good_vars[k] == v) {
				m_good_vars[k] = m_good_vars.back();
				m_good_vars.pop_back();
				break;
			}
		}
	}
}

// Incrementally flip variable var to val: update the current log-probability, the
// affected factor indices, and the per-value score deltas of var and all of its
// Markov-blanket neighbors, then refresh the good-variable set for those vars.
void sls::flip_to(vindex var, size_t val) {
	if (val == m_value[var]) return;

	if (m_task == Task::MMAP) {
		// MMAP: no incremental delta cache -- commit the flip, keep the flipped
		// variable's factor indices current (for GLS+ penalty lookups), recompute
		// the (global) objective via the WMB conditioned evaluator, and refresh
		// the good-variable set over the MAP variables.
		size_t cur = m_value[var];
		for (size_t j = 0; j < m_occ[var].size(); ++j) {
			findex f = m_occ[var][j].first;
			size_t pos = m_occ[var][j].second;
			m_index[f] += (long) m_stride[f][pos] * ((long) val - (long) cur);
		}
		m_value[var] = val;
		++m_num_flips;
		++m_num_flips_total;
		m_log_prob = eval_mmap(m_value);
		for (size_t k = 0; k < m_map_vars.size(); ++k)
			update_good_var(m_map_vars[k]);
		return;
	}

	bool gls = !m_penalty.empty();
	size_t cur = m_value[var];

	for (size_t j = 0; j < m_occ[var].size(); ++j) {
		findex f = m_occ[var][j].first;
		size_t pos = m_occ[var][j].second;
		size_t old_index = m_index[f];
		long shift = (long) m_stride[f][pos] * ((long) val - (long) cur);
		size_t new_index = old_index + shift;

		// Objective change from this factor.
		m_log_prob += m_log_factor[f][new_index] - m_log_factor[f][old_index];

		// The score deltas cached for var itself were computed relative to this
		// factor's old value; shift them so 'val' becomes the new reference (its
		// delta is 0). We recompute var's deltas fully below, so only update the
		// OTHER scope variables here.
		m_index[f] = new_index;

		// Update the per-value score deltas of the other variables in this factor:
		// their delta depends on this factor's current entry, which just moved.
		for (size_t q = 0; q < m_scope[f].size(); ++q) {
			vindex u = m_scope[f][q];
			if (u == var) continue;
			size_t u_stride = m_stride[f][q];
			size_t u_cur = m_value[u];
			// index of the factor entry with u set to value w (others as current):
			//   new_index - u_stride*u_cur + u_stride*w
			size_t u_base = new_index - u_stride * u_cur;      // u set to 0
			size_t u_base_old = old_index - u_stride * u_cur;  // (before the flip)
			for (size_t w = 0; w < m_dom[u]; ++w) {
				if (w == u_cur) continue;
				size_t iw_new = u_base + u_stride * w;
				size_t iw_old = u_base_old + u_stride * w;
				// delta(u->w) contribution from this factor =
				//   log[entry with u=w] - log[entry with u=u_cur] (at current f-index)
				// Remove the old contribution (using old f-index) and add the new.
				double old_contrib = m_log_factor[f][iw_old]
						- m_log_factor[f][old_index];
				double new_contrib = m_log_factor[f][iw_new]
						- m_log_factor[f][new_index];
				m_log_score[u][w] += new_contrib - old_contrib;
				if (gls) {
					double old_p = m_penalty[f][iw_old] - m_penalty[f][old_index];
					double new_p = m_penalty[f][iw_new] - m_penalty[f][new_index];
					m_pen_score[u][w] += new_p - old_p;
				}
			}
		}
	}

	// Commit the flip.
	m_value[var] = val;
	++m_num_flips;
	++m_num_flips_total;

	// Recompute var's own score deltas from scratch (cheap: its occurrences).
	for (size_t val2 = 0; val2 < m_dom[var]; ++val2) {
		double dl = 0.0, dp = 0.0;
		if (val2 != val) {
			for (size_t j = 0; j < m_occ[var].size(); ++j) {
				findex f = m_occ[var][j].first;
				size_t pos = m_occ[var][j].second;
				size_t base = m_index[f];
				long shift = (long) m_stride[f][pos] * ((long) val2 - (long) val);
				dl += m_log_factor[f][base + shift] - m_log_factor[f][base];
				if (gls)
					dp += m_penalty[f][base + shift] - m_penalty[f][base];
			}
		}
		m_log_score[var][val2] = dl;
		if (gls) m_pen_score[var][val2] = dp;
	}

	// Refresh good-variable status for var and its neighbors.
	for (size_t j = 0; j < m_mb[var].size(); ++j)
		update_good_var(m_mb[var][j]);
}

// Best strictly-improving (var,val) over the good variables. Returns false and
// leaves out_var unset if none exists (a local minimum). Random tie-break.
bool sls::best_new_inst(vindex& out_var, size_t& out_val) {
	if (m_good_vars.empty()) return false;
	double best_score = -std::numeric_limits<double>::infinity();
	std::vector<std::pair<vindex, size_t> > best;
	for (size_t k = 0; k < m_good_vars.size(); ++k) {
		vindex v = m_good_vars[k];
		for (size_t val = 0; val < m_dom[v]; ++val) {
			if (val == m_value[v]) continue;
			double s = (m_task == Task::MMAP) ? score_mmap(v, val) : score(v, val);
			if (s > best_score + SLS_EPS) {
				best_score = s;
				best.clear();
				best.push_back(std::make_pair(v, val));
			} else if (s > best_score - SLS_EPS) {
				best.push_back(std::make_pair(v, val));
			}
		}
	}
	if (best.empty() || best_score <= SLS_EPS) return false;
	size_t r = (size_t) (randu() * best.size());
	if (r >= best.size()) r = best.size() - 1;
	out_var = best[r].first;
	out_val = best[r].second;
	return true;
}

// Sample a new value for var proportional to exp(score) (excludes current value).
size_t sls::sampled_new_val(vindex var) {
	if (m_dom[var] <= 1) return m_value[var];
	std::vector<double> w(m_dom[var]);
	double mx = -std::numeric_limits<double>::infinity();
	for (size_t val = 0; val < m_dom[var]; ++val) {
		w[val] = (val == m_value[var]) ? -std::numeric_limits<double>::infinity()
				: ((m_task == Task::MMAP) ? score_mmap(var, val) : score(var, val));
		mx = std::max(mx, w[val]);
	}
	if (mx == -std::numeric_limits<double>::infinity())
		return random_new_val(var); // all equal / degenerate
	double sum = 0.0;
	for (size_t val = 0; val < m_dom[var]; ++val) {
		w[val] = std::exp(w[val] - mx); // softmax (current value -> 0)
		sum += w[val];
	}
	double r = randu() * sum;
	double acc = 0.0;
	for (size_t val = 0; val < m_dom[var]; ++val) {
		acc += w[val];
		if (r <= acc) return val;
	}
	return random_new_val(var);
}

size_t sls::random_new_val(vindex var) {
	if (m_dom[var] <= 1) return m_value[var];
	size_t val = m_value[var];
	while (val == m_value[var]) val = (size_t) randi2((int) m_dom[var]);
	return val;
}

// A random variable the local search may flip: all variables for MAP, only the
// MAP/query variables for MMAP (SUM variables are never flipped).
sls::vindex sls::random_search_var() {
	if (m_task == Task::MMAP)
		return m_map_vars.empty() ? (vindex) 0
				: m_map_vars[(size_t) randi2((int) m_map_vars.size())];
	return (vindex) randi2((int) m_gmo.nvar());
}

void sls::update_best() {
	if (m_log_prob > m_logz + SLS_EPS || !m_found_solution) {
		m_logz = m_log_prob;
		for (size_t v = 0; v < m_gmo.nvar(); ++v) m_best_config[v] = m_value[v];
		m_found_solution = true;
	}
}

void sls::random_restart() {
	if (m_task == Task::MMAP) {
		// Randomize only the MAP variables; SUM variables stay unset (== -1) so the
		// WMB estimator marginalizes over them.
		std::fill(m_value.begin(), m_value.end(), (size_t) -1);
		for (size_t k = 0; k < m_map_vars.size(); ++k) {
			vindex v = m_map_vars[k];
			m_value[v] = (size_t) randi2((int) m_dom[v]);
		}
	} else {
		for (size_t v = 0; v < m_gmo.nvar(); ++v)
			m_value[v] = (size_t) randi2((int) m_dom[v]);
	}
	recompute_scores();
	++m_num_restarts;
}

void sls::init() {
	std::cout << "[SLS] + inference task   : "
			<< (m_task == Task::MMAP ? "MMAP" : "MAP") << std::endl;
	std::cout << "[SLS] + time limit       : " << m_time_limit << std::endl;
	std::cout << "[SLS] + max flips        : " << m_max_flips << std::endl;
	std::cout << "[SLS] + noise (%)        : " << m_noise << std::endl;
	std::cout << "[SLS] + seed             : " << m_seed << std::endl;
	if (m_task == Task::MMAP)
		std::cout << "[SLS] + i-bound (MMAP)   : " << m_ibound << std::endl;

	// A large finite clamp for log(0): well below any realistic log-probability
	// so a zero-probability assignment is strongly disfavored but never -inf.
	LOG_ZERO = -1.0e6;

	rand_seed(m_seed);
	build_structures();

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

// The G+StS main loop: greedy best flip with an occasional noisy random walk,
// restarting when a run stagnates (spends more than cutoff x its time-to-best
// without improving). Runs until the time/flip budget is exhausted.
void sls::run() {
	m_start_time = timeSystem();
	init();

	// Initial random assignment + score seed.
	random_restart();
	m_num_restarts = 0; // the initial "restart" doesn't count
	update_best();

	double last_cutoff_time = 0.0;
	double best_this_restart = m_log_prob;
	double best_time_this_restart = 0.0;
	long recompute_interval = 100000;

	while (ls_continue()) {
		// Periodic full recompute to cancel floating-point drift.
		if (m_num_flips > 0 && (m_num_flips % recompute_interval) == 0)
			recompute_scores();

		vindex var = (vindex) -1;
		size_t val = 0;
		if ((randu() * 100.0) < (double) m_noise) {
			// Noisy step: random (MAP) variable, sampled value.
			var = random_search_var();
			val = sampled_new_val(var);
		} else {
			// Greedy step: best improving flip.
			if (!best_new_inst(var, val)) {
				// Local minimum: G+StS takes no informative move here; count a
				// flip so a pure flip budget still terminates, and force a noisy
				// move to escape.
				var = random_search_var();
				val = sampled_new_val(var);
			}
		}
		if (var != (vindex) -1 && val != m_value[var])
			flip_to(var, val);
		else
			++m_num_flips; // degenerate (e.g. singleton domain); avoid infinite loop

		update_best();

		if (m_log_prob > best_this_restart) {
			best_this_restart = m_log_prob;
			best_time_this_restart = (timeSystem() - m_start_time) - last_cutoff_time;
		}
		double run_time_this_restart =
				(timeSystem() - m_start_time) - last_cutoff_time;
		if (m_time_limit > 0.0 &&
				run_time_this_restart > std::max(m_cutoff * best_time_this_restart, 0.1)) {
			random_restart();
			last_cutoff_time = timeSystem() - m_start_time;
			best_time_this_restart = 0.0;
			best_this_restart = m_log_prob;
			update_best();
		}
	}

	m_logz = m_found_solution ? m_logz : -std::numeric_limits<double>::infinity();

	std::cout << "[SLS] Finished searching in "
			<< (timeSystem() - m_start_time) << " seconds" << std::endl;
	std::cout << "[SLS] + flips            : " << m_num_flips_total << std::endl;
	std::cout << "[SLS] + restarts         : " << m_num_restarts << std::endl;
	std::cout << "[SLS] Best value found   : " << m_logz
			<< " (" << std::exp(m_logz) << ")" << std::endl;
}

void sls::write_solution(std::ostream& out, const std::map<size_t, size_t>& evidence,
		const std::map<size_t, size_t>& old2new, const graphical_model& orig,
		const std::set<size_t>& dummies, int output_format) {
	// Local search never proves optimality.
	const char* status = "false";

	const bool mmap = (m_task == Task::MMAP);
	const char* task_name = mmap ? "MMAP" : "MAP";

	if (output_format == MERLIN_OUTPUT_JSON) {
		out << "{";
		out << " \"algorithm\" : \"" << algo_name() << "\", ";
		out << " \"seed\" : " << m_seed << ", ";
		out << " \"task\" : \"" << task_name << "\", ";
		out << " \"value\" : " << std::fixed << std::setprecision(MERLIN_PRECISION)
			<< (m_logz + std::log(orig.get_global_const())) << ", ";
		out << " \"status\" : \"" << status << "\", ";
		out << " \"optimal\" : false, ";
		out << " \"solution\" : [ ";
		if (mmap) {
			// MMAP: report only the query (MAP) variables (post-evidence indices).
			for (vindex i = 0; i < m_query.size(); ++i) {
				vindex j = m_query[i];
				out << "{ \"variable\" : " << j << ", \"value\" : "
					<< m_best_config[j] << "}";
				if (i != m_query.size() - 1) out << ", ";
			}
		} else {
			for (vindex i = 0; i < orig.nvar(); ++i) {
				if (dummies.find(i) != dummies.end()) continue;
				out << "{";
				out << " \"variable\" : " << i << ",";
				try {
					size_t val = evidence.at(i);
					out << " \"value\" : " << val;
				} catch (std::out_of_range& e) {
					vindex j = old2new.at(i);
					out << " \"value\" : " << m_best_config[j];
				}
				out << "}";
				if (i != orig.nvar() - 1) out << ", ";
			}
		}
		out << "] }";
	} else if (output_format == MERLIN_OUTPUT_UAI) {
		if (mmap) {
			out << "MMAP" << std::endl;
			out << m_query.size();
			for (vindex i = 0; i < m_query.size(); ++i)
				out << " " << m_best_config[m_query[i]];
			out << std::endl;
		} else {
			out << "MAP" << std::endl;
			out << orig.nvar() - dummies.size();
			for (vindex i = 0; i < orig.nvar(); ++i) {
				if (dummies.find(i) != dummies.end()) continue;
				try {
					size_t val = evidence.at(i);
					out << " " << val;
				} catch (std::out_of_range& e) {
					vindex j = old2new.at(i);
					out << " " << m_best_config[j];
				}
			}
			out << std::endl;
		}
	} else {
		throw std::runtime_error("Unknown output format.");
	}
}

} // namespace merlin
