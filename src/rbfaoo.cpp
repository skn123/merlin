/*
 * rbfaoo.cpp
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

/// \file rbfaoo.cpp
/// \brief Recursive AND/OR Best-First search with Overestimation (RBFAOO)
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#include "rbfaoo.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace merlin {

namespace {
const double RB_INF = std::numeric_limits<double>::infinity();
const double RB_EPS = 1e-10; // value-threshold slack (matches ground truth)
// Convert a linear-space probability to cost (negative-log). Zero -> +inf.
inline double to_cost(double p) {
	return (p <= 0.0) ? RB_INF : -std::log(p);
}
} // anonymous namespace

// Record an AND node's assignment on the current path (OR nodes: no-op).
void rbfaoo::do_process(rbfaoo_node* n) {
	if (n->type() == AO_AND)
		m_assignment[n->var()] = n->val();
}

// Compute the (weighted, absolute) cost heuristic of an OR node and cache the
// per-value (total cost incl. arc, arc cost) pairs used to expand its AND
// children. Cost space:
//   arc[v]   = label(var=v)                              (= -log of the label)
//   total[v] = m_epsilon * (-log(get_heuristic) - subtree_norm[var]) + arc[v]
//   OR heur  = min over v (MAX var) or -logsumexp over v (SUM var).
// The absolute WMB bound at a SUM variable already upper-bounds the summed
// completion below it, so the aggregate is an admissible cost lower bound for
// BOTH MAX and SUM nodes -- essential for best-first correctness (the SUM
// heuristic must NOT be zeroed, or a MAX parent can commit prematurely).
double rbfaoo::heuristic_or_cost(rbfaoo_node& n) {
	size_t var = n.var();
	bool map_var = m_var_types[var];
	size_t dom = m_gmo.var(var).states();
	double offset = m_subtree_norm[var];
	double* dv = new double[dom * 2];
	double h = map_var ? RB_INF : 0.0; // MAX: min accumulator; SUM: logsumexp
	double sum_min = RB_INF;
	size_t saved = m_assignment[var];
	for (size_t i = 0; i < dom; ++i) {
		m_assignment[var] = i;
		double lbl = label(var, m_assignment); // arc cost (+inf if zero prob)
		double arc = lbl;
		double hv = m_heuristic.get_heuristic(var, m_assignment); // linear UB
		double comp = (hv <= 0.0) ? RB_INF : (m_epsilon * (-std::log(hv) - offset));
		double total = (arc == RB_INF) ? RB_INF : (comp + arc);
		dv[2 * i + 1] = arc;   // arc/label cost
		dv[2 * i] = total;     // total heuristic cost (includes arc)
		if (map_var)
			h = std::min(h, total);
		else
			sum_min = std::min(sum_min, total);
	}
	m_assignment[var] = saved;
	if (!map_var) { // SUM: h = -logsumexp(-total) over values
		if (sum_min == RB_INF) {
			h = RB_INF;
		} else {
			double s = 0.0;
			for (size_t i = 0; i < dom; ++i)
				s += std::exp(sum_min - dv[2 * i]);
			h = sum_min - std::log(s);
		}
	}
	n.set_heur(h);
	n.set_heur_cache(dv);
	return h;
}

// Generate the OR children (one AND child per domain value) of an AND node, read
// their cached values/dns if present, and aggregate: AND value = sum of children
// costs, dn = min child dn (best child), tracking second_best_dn. Solved when
// the aggregate dn is DNINF.
int rbfaoo::generate_children_and(rbfaoo_node& n, int& second_best_dn,
		int& best_index, int& solved_flag) {
	size_t var = n.var();
	int depth = n.depth();

	double value = 0.0; // AND = sum of children (cost)
	int dn = DNINF;
	second_best_dn = DNINF;
	best_index = UNDETERMINED;
	solved_flag = 0;

	int num_children = 0;
	// The dummy super-root (var == -1) branches over the pseudo-tree roots; a real
	// AND node branches over its variable's pseudo-tree children.
	const std::vector<vindex>& kids = (var == (size_t) -1) ? m_roots : m_children[var];
	for (size_t k = 0; k < kids.size(); ++k, ++num_children) {
		size_t vchild = kids[k];
		rbfaoo_node* c = new rbfaoo_node(AO_OR, vchild, depth + 1);
		// OR child context = its cache-context variables' current values.
		const variable_set& ctx = m_cache_context[vchild];
		for (variable_set::const_iterator it = ctx.begin(); it != ctx.end(); ++it)
			c->add_context(m_assignment[it->label()]);
		c->zobrist().encodeOR(ctx, m_assignment);

		RbfaooCacheElementPtr p = m_dfpncache->read(c->zobrist(), (int) vchild, 1,
				c->context());
		double child_value;
		int child_dn;
		if (p == NULL) {
			child_value = heuristic_or_cost(*c);       // absolute cost bound (MAX or SUM)
			child_dn = 1;
		} else {
			heuristic_or_cost(*c);                      // seed heur cache (for reuse)
			child_value = p->result;                    // edge cost 0 from AND to OR
			child_dn = p->dn;
		}
		c->set_node_value(child_value);
		c->set_dn(child_dn);
		value += child_value;
		if (child_dn < dn) {
			second_best_dn = dn;
			dn = child_dn;
			best_index = num_children;
		} else if (child_dn < second_best_dn) {
			second_best_dn = child_dn;
		}
		m_expand.push_back(c);
	}

	// A +inf cost means an infeasible child: the whole AND sub-problem is a proven
	// dead end -- mark solved with a terminal dn so the parent stops re-expanding.
	if (value == RB_INF) dn = DNINF;
	n.set_node_value(value);
	n.set_dn(dn);
	if (dn == DNINF) solved_flag = 1;
	return num_children;
}

// Recompute an AND node's value/dn from its (cached) OR children.
void rbfaoo::calculate_and(rbfaoo_node& n, size_t table_index, int num_children,
		int& second_best_dn, int& best_index, int& solved_flag) {
	double value = 0.0;
	int dn = DNINF;
	second_best_dn = DNINF;
	best_index = UNDETERMINED;
	solved_flag = 0;
	for (size_t i = 0; i < (size_t) num_children; ++i) {
		rbfaoo_node* c = m_expand[table_index + i];
		RbfaooCacheElementPtr p = m_dfpncache->read(c->zobrist(),
				(int) c->var(), 1, c->context());
		double child_value;
		int child_dn;
		if (p == NULL) {
			child_value = c->heur();
			child_dn = 1;
		} else {
			child_value = p->result;
			child_dn = p->dn;
		}
		c->set_node_value(child_value);
		c->set_dn(child_dn);
		value += child_value;
		if (child_dn < dn) {
			second_best_dn = dn;
			dn = child_dn;
			best_index = i;
		} else if (child_dn < second_best_dn) {
			second_best_dn = child_dn;
		}
	}
	// A +inf cost = infeasible child => proven dead-end AND sub-problem.
	if (value == RB_INF) dn = DNINF;
	n.set_node_value(value);
	n.set_dn(dn);
	if (dn == DNINF) solved_flag = 1;
}

// Generate the AND children (one per domain value) of an OR node, read their
// cached values/dns, and aggregate. MAX var: value = min child cost, tracking
// second-best and the cutoff (min over proven children). SUM var: value =
// -logsumexp(-child cost), computed inline. best_index == the chosen value.
int rbfaoo::generate_children_or(rbfaoo_node& n, double& second_best_value,
		int& best_index, int& solved_flag, double& cutoff_value) {
	size_t var = n.var();
	bool map_var = m_var_types[var];
	int depth = n.depth();
	solved_flag = map_var ? 0 : 1;

	double* heur = n.heur_cache();
	double value = map_var ? RB_INF : 0.0;
	cutoff_value = second_best_value = RB_INF;
	int dn = 0, dn_ignoring_proven = 0;
	best_index = UNDETERMINED;
	size_t dom = m_gmo.var(var).states();

	int num_children = 0;
	for (size_t i = 0; i < dom; ++i, ++num_children) {
		rbfaoo_node* c = new rbfaoo_node(AO_AND, var, i, depth, heur[2 * i + 1]);
		c->zobrist().encodeAND(n.zobrist(), i);
		c->set_context(n.context());
		c->set_heur(heur[2 * i]);
		RbfaooCacheElementPtr p = m_dfpncache->read(c->zobrist(), (int) var, 0,
				c->context());
		double child_value;
		int child_dn, child_solved;
		if (p == NULL) {
			child_value = heur[2 * i];
			child_dn = 1;
			child_solved = 0;
		} else {
			child_value = p->result + c->label(); // add the OR->AND arc cost
			child_dn = p->dn;
			child_solved = p->solved_flag;
		}
		if (heur[2 * i] == RB_INF) {
			child_value = RB_INF;
			child_dn = 0; // dead end: cannot be part of a finite solution
		}

		if (map_var) {
			if (child_value < value) {
				best_index = i;
				second_best_value = value;
				value = child_value;
				solved_flag = child_solved;
			} else if (child_value == value && child_solved) {
				best_index = i;
				second_best_value = value;
				solved_flag = child_solved;
			} else if (child_value < second_best_value) {
				second_best_value = child_value;
			}
			if (child_dn != DNINF) {
				dn_ignoring_proven = sum_dn(dn_ignoring_proven, child_dn);
			} else {
				cutoff_value = std::min(cutoff_value, child_value);
			}
			dn = sum_dn(dn, child_dn);
		} else { // SUM var: accumulate exp(-cost) for logsumexp
			value += std::exp(-child_value);
			best_index = i;
			second_best_value = value;
			solved_flag &= child_solved;
			if (child_dn != DNINF)
				dn_ignoring_proven = sum_dn(dn_ignoring_proven, child_dn);
			else
				cutoff_value = RB_INF;
			dn = sum_dn(dn, child_dn);
		}

		c->set_node_value(child_value);
		c->set_dn(child_dn);
		m_expand.push_back(c);
	}

	if (!solved_flag && dn == DNINF)
		dn = std::max(1, dn_ignoring_proven);
	if (!map_var)
		value = to_cost(value); // -log(sum exp(-cost)) = logsumexp in cost space

	// A +inf cost is a proven dead end (zero-probability sub-problem): mark it
	// solved with a terminal dn so the parent stops re-expanding it (else the
	// DFPN recursion never terminates on an infeasible sub-problem).
	if (value == RB_INF) { solved_flag = 1; dn = DNINF; }

	n.set_node_value(value);
	n.set_dn(dn);
	return num_children;
}

// Recompute an OR node's value/dn from its (cached) AND children.
void rbfaoo::calculate_or(rbfaoo_node& n, size_t table_index, int num_children,
		double& second_best_value, int& best_index, int& solved_flag,
		double& cutoff_value) {
	size_t var = n.var();
	bool map_var = m_var_types[var];
	solved_flag = map_var ? 0 : 1;
	double* heur = n.heur_cache();
	double value = map_var ? RB_INF : 0.0;
	cutoff_value = second_best_value = RB_INF;
	int dn = 0, dn_ignoring_proven = 0;
	best_index = UNDETERMINED;

	for (int i = 0; i < num_children; ++i) {
		rbfaoo_node* c = m_expand[table_index + i];
		RbfaooCacheElementPtr p = m_dfpncache->read(c->zobrist(),
				(int) c->var(), 0, c->context());
		double child_value;
		int child_dn, child_solved;
		if (p == NULL) {
			if (heur[2 * i] == RB_INF) {
				child_value = RB_INF;
				child_dn = 0;
			} else {
				child_value = c->heur();
				child_dn = 1;
			}
			child_solved = 0;
		} else {
			child_value = p->result + c->label();
			child_dn = p->dn;
			child_solved = p->solved_flag;
		}

		if (map_var) {
			if (child_value < value) {
				best_index = i;
				second_best_value = value;
				value = child_value;
				solved_flag = child_solved;
			} else if (child_value == value) {
				if (child_solved) {
					best_index = i;
					second_best_value = value;
					solved_flag = child_solved;
				}
			} else if (child_value < second_best_value) {
				second_best_value = child_value;
			}
			if (child_dn != DNINF)
				dn_ignoring_proven = sum_dn(dn_ignoring_proven, child_dn);
			else
				cutoff_value = std::min(cutoff_value, child_value);
			dn = sum_dn(dn, child_dn);
		} else { // SUM var
			value += std::exp(-child_value);
			second_best_value = value;
			solved_flag &= child_solved;
			if (!child_solved) best_index = i;
			if (child_dn != DNINF)
				dn_ignoring_proven = sum_dn(dn_ignoring_proven, child_dn);
			else
				cutoff_value = RB_INF;
			dn = sum_dn(dn, child_dn);
		}
		c->set_node_value(child_value);
		c->set_dn(child_dn);
	}

	if (!solved_flag && dn == DNINF)
		dn = std::max(1, dn_ignoring_proven);
	if (!map_var)
		value = to_cost(value);
	// A +inf cost is a proven dead end: mark solved so the parent stops retrying.
	if (value == RB_INF) { solved_flag = 1; dn = DNINF; }
	n.set_node_value(value);
	n.set_dn(dn);
}

// Mutually-recursive iterative deepening on an OR node (cost = min over MAX
// children / logsumexp over SUM children). Recurse into the best child until the
// node's value exceeds th_value or it is solved / disproven.
void rbfaoo::mid_or(rbfaoo_node& n, size_t table_index, double th_value,
		int dn_threshold) {
	bool generated = false;
	int num_children = 0, best_index = UNDETERMINED, solved_flag = 0;
	size_t var = n.var();
	rbfaoo_context_t ctxt = n.context();
	size_t increased = m_num_expanded;
	double value;
	int dn;

	for (;;) {
		double second_best_value, cutoff_value;
		if (!generated) {
			num_children = generate_children_or(n, second_best_value, best_index,
					solved_flag, cutoff_value);
			++m_num_expanded;
			++m_num_expanded_or;
		} else {
			calculate_or(n, table_index, num_children, second_best_value,
					best_index, solved_flag, cutoff_value);
		}
		generated = true;
		value = n.node_value();
		dn = n.dn();

		// Stop when the value exceeds the threshold, the node is solved, or no
		// viable best child exists (best_index < 0 => every child is +inf cost, a
		// zero-probability dead end -- avoids dereferencing m_expand[-1]).
		if (value > th_value + RB_EPS || dn == DNINF || solved_flag
				|| best_index < 0)
			break;

		rbfaoo_node* c = m_expand[table_index + best_index];
		double new_th_value;
		if (m_var_types[c->var()]) { // MAP AND child
			new_th_value = std::min(th_value, second_best_value + m_overestimation);
			new_th_value = std::min(new_th_value, cutoff_value);
			// Guard inf - inf (a dead-end best child under an infinite threshold).
			if (new_th_value != RB_INF)
				new_th_value -= c->label();
		} else {
			new_th_value = RB_INF; // SUM subtree: solve exactly in one descent
		}
		int new_dn_threshold = std::min((int) DNINF, dn_threshold - dn + c->dn());
		size_t new_table_index = table_index + num_children;

		if (m_time_limit > 0.0 &&
				(timeSystem() - m_start_time) >= m_time_limit)
			throw std::runtime_error("__rbfaoo_timeout__");

		do_process(c);
		mid_and(*c, new_table_index, new_th_value, new_dn_threshold);
	}

	increased = m_num_expanded - increased;
	for (size_t i = table_index; i < table_index + (size_t) num_children; ++i)
		delete m_expand[i];
	m_expand.resize(table_index);
	m_dfpncache->write(n.zobrist(), (int) var, ctxt, 1, best_index, value, dn,
			solved_flag, (unsigned int) increased);
	if (!m_var_types[var]) ++m_num_sum_evals;
}

// Mutually-recursive iterative deepening on an AND node (cost = sum of OR
// children). Recurse into the min-dn child until value exceeds th_value or the
// node is solved / disproven.
void rbfaoo::mid_and(rbfaoo_node& n, size_t table_index, double th_value,
		int dn_threshold) {
	bool generated = false;
	int num_children = 0, best_index = UNDETERMINED, solved_flag = 0;
	size_t var = n.var();
	rbfaoo_context_t ctxt = n.context();
	size_t increased = m_num_expanded;
	double value;
	int dn;

	for (;;) {
		int second_best_dn;
		if (!generated) {
			num_children = generate_children_and(n, second_best_dn, best_index,
					solved_flag);
			++m_num_expanded;
			++m_num_expanded_and;
		} else {
			calculate_and(n, table_index, num_children, second_best_dn,
					best_index, solved_flag);
		}
		generated = true;
		value = n.node_value();
		dn = n.dn();

		// Stop when the value exceeds the threshold, the node is solved, or no
		// viable best child exists (best_index < 0 => a zero-probability dead end).
		if (value > th_value + RB_EPS || dn == DNINF || solved_flag
				|| best_index < 0)
			break;

		rbfaoo_node* c = m_expand[table_index + best_index];
		double c_value = c->node_value();
		double new_th_value;
		if (m_var_types[c->var()]) { // MAP OR child
			// new_th = (th - value) + c_value. Guard inf - inf (dead node under an
			// infinite threshold) which would otherwise yield NaN.
			if (th_value == RB_INF || value == RB_INF)
				new_th_value = RB_INF;
			else
				new_th_value = (th_value - value) + c_value;
		} else {
			new_th_value = RB_INF;
		}
		int new_dn_threshold = std::min(sum_dn(second_best_dn, 1), dn_threshold);
		size_t new_table_index = table_index + num_children;

		if (m_time_limit > 0.0 &&
				(timeSystem() - m_start_time) >= m_time_limit)
			throw std::runtime_error("__rbfaoo_timeout__");

		mid_or(*c, new_table_index, new_th_value, new_dn_threshold);
	}

	increased = m_num_expanded - increased;
	for (size_t i = table_index; i < table_index + (size_t) num_children; ++i)
		delete m_expand[i];
	m_expand.resize(table_index);
	m_dfpncache->write(n.zobrist(), (int) var, ctxt, 0, best_index, value, dn,
			solved_flag, (unsigned int) increased);
}

int rbfaoo::rbfs() {
	m_assignment.assign(m_gmo.nvar(), (size_t) -1);

	// Dummy super-root AND node (var == -1): its OR children are the pseudo-tree
	// roots. The global constant is folded into its label (cost). This mirrors
	// the ground truth's dummy root variable and handles multiple MAP roots.
	double root_label = to_cost(m_gmo.get_global_const());
	if (root_label == RB_INF) root_label = 0.0;
	rbfaoo_node root(AO_AND, (size_t) -1, 0, -1, root_label);
	// Root context is empty; encode from an all-unset assignment.
	Zobrist z;
	z.encodeOR(variable_set(), m_assignment);
	root.zobrist().encodeAND(z, 0);

	int res = 0;
	try {
		mid_and(root, 0, RB_INF, DNINF);
		RbfaooCacheElementPtr p = m_dfpncache->read(root.zobrist(), -1, 0,
				root.context());
		m_root_result = (p != NULL) ? p->result : RB_INF;
	} catch (std::runtime_error&) {
		res = 1; // timeout
		RbfaooCacheElementPtr p = m_dfpncache->read(root.zobrist(), -1, 0,
				root.context());
		m_root_result = (p != NULL) ? p->result : RB_INF;
	}
	return res;
}

// Reconstruct the MAP/MMAP assignment by descending the transposition table from
// the root, following each MAP OR node's cached best_index (== the chosen value).
// SUM variables are marginalized (no value in the output), so we only descend
// their subtrees to keep the context keys consistent for deeper MAP variables.
void rbfaoo::reconstruct_assignment() {
	size_t n = m_gmo.nvar();
	std::fill(m_best_config.begin(), m_best_config.end(), 0);
	std::vector<size_t> asgn(n, (size_t) -1);

	// Iterative descent over OR variables in pseudo-tree order. Start from the
	// roots; for each OR variable, read its cached best value, assign it, and
	// enqueue its pseudo-tree children.
	std::vector<size_t> stack(m_roots.begin(), m_roots.end());
	while (!stack.empty()) {
		size_t var = stack.back();
		stack.pop_back();

		// Build the OR node's Zobrist/context from the current partial assignment.
		Zobrist z;
		const variable_set& ctx = m_cache_context[var];
		z.encodeOR(ctx, asgn);
		rbfaoo_context_t ctxt;
		for (variable_set::const_iterator it = ctx.begin(); it != ctx.end(); ++it)
			ctxt.push_back(asgn[it->label()]);

		RbfaooCacheElementPtr p = m_dfpncache->read(z, (int) var, 1, ctxt);
		size_t chosen = 0;
		if (p != NULL && p->best_index >= 0)
			chosen = (size_t) p->best_index; // best_index == chosen value
		asgn[var] = chosen;
		if (m_var_types[var])
			m_best_config[var] = chosen;

		const std::vector<vindex>& kids = m_children[var];
		for (size_t k = 0; k < kids.size(); ++k)
			stack.push_back(kids[k]);
	}
}

void rbfaoo::run() {
	m_start_time = timeSystem();
	init();

	std::cout << "[RBFAOO] + weight (epsilon) : " << m_epsilon << std::endl;
	std::cout << "[RBFAOO] + overestimation   : " << m_overestimation << std::endl;
	std::cout << "[RBFAOO] + cache size (KB)  : " << m_cache_size << std::endl;

	// AND-node context per variable = OR context (ancestors) plus the variable.
	size_t nv = m_gmo.nvar();
	m_and_context.assign(nv, variable_set());
	for (size_t v = 0; v < nv; ++v)
		if (m_cache_context[v].num_states() > 0)
			m_and_context[v] = m_cache_context[v] | m_gmo.var(v);


	// Average context size (for sizing the cache elements' context storage).
	size_t ctx_size = 0;
	for (size_t v = 0; v < nv; ++v) ctx_size = std::max(ctx_size, m_cache_context[v].nvar());

	// Zobrist tables + transposition table.
	Zobrist::init_once(m_gmo);
	m_dfpncache = new RbfaooCacheTable();
	m_dfpncache->init(m_cache_size, ctx_size);

	m_expand.clear();
	m_num_expanded = 0;
	m_num_expanded_or = 0;
	m_num_expanded_and = 0;
	m_num_sum_evals = 0;
	m_root_result = RB_INF;

	int status = rbfs();
	bool timed_out = (status != 0);
	m_found_solution = (m_root_result < RB_INF);
	m_proved_optimal = !timed_out && m_found_solution;

	if (m_found_solution)
		reconstruct_assignment();

	// Report value in log space (log-prob = -cost). No solution => -inf.
	m_incumbent_cost = m_root_result;
	m_logz = m_found_solution ? -m_root_result
			: -std::numeric_limits<double>::infinity();

	std::cout << "[RBFAOO] Finished searching in "
			<< (timeSystem() - m_start_time) << " seconds" << std::endl;
	std::cout << "[RBFAOO] + OR nodes         : " << m_num_expanded_or << std::endl;
	std::cout << "[RBFAOO] + AND nodes        : " << m_num_expanded_and << std::endl;
	if (m_task == Task::MMAP)
		std::cout << "[RBFAOO] + SUM evaluations  : " << m_num_sum_evals << std::endl;
	if (timed_out)
		std::cout << "[RBFAOO] + status           : time limit reached (suboptimal)"
				<< std::endl;
	if (!m_found_solution)
		std::cout << "[RBFAOO] + status           : no complete solution found within "
				"the time limit" << std::endl;
	std::cout << "[RBFAOO] " << (m_proved_optimal ? "Optimal value      : " : "Best value so far  : ")
			<< m_logz << " (" << std::exp(m_logz) << ")" << std::endl;

	// Clean up.
	m_expand.clear();
	delete m_dfpncache;
	m_dfpncache = NULL;
	Zobrist::finish_once();
}

} // namespace merlin
