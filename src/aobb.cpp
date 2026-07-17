/*
 * aobb.cpp
 *
 *  Created on: 15 Jul 2026
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

/// \file aobb.cpp
/// \brief AND/OR Branch-and-Bound (AOBB) search with weighted mini-bucket heuristics
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#include "aobb.h"
#include "gls.h"

#include <algorithm>
#include <set>
#include <stack>
#include <sstream>

namespace merlin {

// Product of the original factors anchored at 'var' (those whose deepest
// pseudo-tree/elimination variable is 'var'), evaluated at the current
// assignment. All scope variables of such factors are already instantiated.
double aobb::label(vindex var, const std::vector<size_t>& asgn) const {
	double lbl = 1.0;
	const flist& fs = m_anchored[var];
	for (flist::const_iterator it = fs.begin(); it != fs.end(); ++it) {
		const factor& f = m_gmo.get_factor(*it);
		const variable_set& sc = f.vars();
		// Build the sub-index into f's table from the current assignment.
		size_t idx = 0, stride = 1;
		for (variable_set::const_iterator vi = sc.begin(); vi != sc.end(); ++vi) {
			vindex v = vi->label();
			// A scope variable may be unset (== -1) if it was omitted from the
			// elimination order (e.g. a cardinality-1 variable): treat it as
			// value 0, which is exact for singleton domains and avoids an index
			// underflow into the factor table.
			size_t vv = (asgn[v] == (size_t) -1) ? 0 : asgn[v];
			idx += stride * vv;
			stride *= vi->states();
		}
		lbl *= f.get(idx);
	}
	// Cost space: -log(product of factors). Zero probability => +infinity.
	return (lbl <= 0.0) ? std::numeric_limits<double>::infinity() : -std::log(lbl);
}

// Lower bound on the COST of the best complete solution consistent with the
// partial path from the root down through node 'n' (negative-log space). Walk up
// to the root, combining the committed labels (AND) and the sub-tree cost bounds
// of siblings (solved cost where available, heuristic otherwise) with the
// AND=sum, OR=min (MAX var) / logsumexp (SUM var) rule. This is the native
// equivalent of AOBB's canBePruned; prune when the bound >= incumbent cost.
double aobb::path_bound(ao_node* n) const {
	// Cost contributed by the sub-tree rooted at 'n' itself.
	double acc = (n->type == AO_AND) ? (n->label + n->heur) : n->heur;

	ao_node* cur = n;
	ao_node* par = n->parent;
	while (par != NULL) {
		if (par->type == AO_AND) {
			// AND parent: add its label cost and the sibling OR nodes'
			// contributions (solved cost if available, else heuristic).
			acc += par->label;
			for (size_t i = 0; i < par->children.size(); ++i) {
				ao_node* c = par->children[i];
				if (c == cur) continue;
				acc += (c->solved ? c->value : c->heur);
			}
		} else if (par->is_max_or) {
			// OR parent over a MAX var: the best sibling branch is the min cost;
			// a lower bound on this OR node's cost is the min over all branches.
			for (size_t i = 0; i < par->children.size(); ++i) {
				ao_node* c = par->children[i];
				if (c == cur) continue;
				double cv = (c->solved ? c->value : c->heur);
				acc = std::min(acc, cv);
			}
		} else {
			// OR parent over a SUM var: combine by -logsumexp(-cost) over all
			// branches (this branch's 'acc' plus the sibling bounds).
			double mn = acc;
			for (size_t i = 0; i < par->children.size(); ++i) {
				ao_node* c = par->children[i];
				if (c == cur) continue;
				mn = std::min(mn, (c->solved ? c->value : c->heur));
			}
			if (mn == std::numeric_limits<double>::infinity()) {
				// Every branch is a zero-probability dead end: cost stays +inf.
				// (Avoids exp(inf-inf)=NaN in the logsumexp below.)
				acc = std::numeric_limits<double>::infinity();
			} else {
				double sum = std::exp(acc - mn); // this branch
				for (size_t i = 0; i < par->children.size(); ++i) {
					ao_node* c = par->children[i];
					if (c == cur) continue;
					double cv = (c->solved ? c->value : c->heur);
					sum += std::exp(mn - cv);
				}
				acc = mn - std::log(sum);
			}
		}
		cur = par;
		par = par->parent;
	}
	return acc;
}

void aobb::free_subtree(ao_node* n) {
	if (n == NULL) return;
	for (size_t i = 0; i < n->children.size(); ++i) {
		free_subtree(n->children[i]);
	}
	delete n;
}

void aobb::init() {

	// Prologue
	std::cout << "[AOBB] + i-bound          : " << m_ibound << std::endl;
	std::cout << "[AOBB] + iterations       : " << m_num_iter << std::endl;
	std::cout << "[AOBB] + inference task   : "
			<< (m_task == Task::MAP ? "MAP" : "MMAP") << std::endl;
	std::cout << "[AOBB] + ordering heur.   : " << m_order_method << std::endl;
	std::cout << "[AOBB] + caching          : " << (m_caching ? "Yes" : "No") << std::endl;

	size_t n = m_gmo.nvar();

	// Variable types: true = MAX (query), false = SUM.
	m_var_types.resize(n, false);
	if (m_task == Task::MAP) {
		for (size_t i = 0; i < n; ++i) m_var_types[i] = true; // all variables are MAX
	} else { // MMAP
		std::fill(m_var_types.begin(), m_var_types.end(), false);
		for (size_t i = 0; i < m_query.size(); ++i) m_var_types[m_query[i]] = true;
	}

	// Constrained elimination order (SUM variables eliminated first, so the
	// query/MAX variables end up at the top of the pseudo tree).
	if (m_order.size() == 0) {
		m_order = m_gmo.order2(m_order_method, m_var_types);
	}
	if (m_parents.size() == 0) {
		m_parents = m_gmo.pseudo_tree(m_order);
	}

	std::cout << "[AOBB] + elimination      : ";
	std::copy(m_order.begin(), m_order.end(),
			std::ostream_iterator<size_t>(std::cout, " "));
	std::cout << std::endl;
	size_t wstar = m_gmo.induced_width(m_order);
	std::cout << "[AOBB] + induced width    : " << wstar << std::endl;

	// Pseudo-tree children and roots (parents[v] is later in the order than v;
	// a root has no parent and is explored first / sits at the top).
	m_children.assign(n, std::vector<vindex>());
	m_roots.clear();
	for (size_t v = 0; v < n; ++v) {
		vindex p = m_parents[v];
		if (p == (vindex) -1) {
			m_roots.push_back(v);
		} else {
			m_children[p].push_back(v);
		}
	}

	// Position (elimination rank) of each variable.
	m_position.assign(n, 0);
	for (size_t i = 0; i < m_order.size(); ++i) {
		m_position[m_order[i]] = i;
	}

	// Anchor each factor at the variable in its scope eliminated FIRST (smallest
	// position). In the pseudo tree parents are later in the order, so descending
	// a root->leaf path assigns variables from largest position (root) to
	// smallest (leaf); the factor becomes fully instantiated when the last of
	// its scope variables is assigned, i.e. the one with the smallest position.
	m_anchored.assign(n, flist());
	for (size_t f = 0; f < m_gmo.num_factors(); ++f) {
		const variable_set& sc = m_gmo.get_factor(f).vars();
		if (sc.nvar() == 0) continue; // constant factor: folded into global const
		vindex anchor = sc.begin()->label();
		size_t best_pos = m_position[anchor];
		for (variable_set::const_iterator vi = sc.begin(); vi != sc.end(); ++vi) {
			vindex v = vi->label();
			if (m_position[v] <= best_pos) {
				best_pos = m_position[v];
				anchor = v;
			}
		}
		m_anchored[anchor] |= f;
	}

	// Compute the OR cache context for each variable. The context of x is the
	// set of ancestors of x (in the pseudo tree) that appear in the scope of
	// some factor touching x's sub-tree (x or a descendant). Two OR nodes for x
	// with the same context assignment have identical optimal sub-tree values,
	// so the value can be cached and reused. (Used only when m_caching is on.)
	m_cache_context.assign(n, variable_set());
	if (m_caching) {
		// depth of each variable and ancestor sets (via m_parents)
		std::vector<variable_set> ancestors(n);
		for (size_t oi = m_order.size(); oi-- > 0; ) {
			// process from roots (end of order) down to leaves (start of order)
			vindex v = m_order[oi];
			vindex p = m_parents[v];
			if (p != (vindex) -1) {
				ancestors[v] = ancestors[p];
				ancestors[v] |= m_gmo.var(p);
			}
		}
		// union of factor scopes touching x's subtree, intersected with ancestors
		// scope_union[x] accumulates scopes of factors anchored in x's subtree
		std::vector<variable_set> scope_union(n);
		for (size_t f = 0; f < m_gmo.num_factors(); ++f) {
			const variable_set& sc = m_gmo.get_factor(f).vars();
			if (sc.nvar() == 0) continue;
			// the deepest (smallest-position) scope var is where the factor is
			// anchored; add its scope to every ancestor-or-self along the path.
			vindex deep = sc.begin()->label();
			size_t best_pos = m_position[deep];
			for (variable_set::const_iterator vi = sc.begin(); vi != sc.end(); ++vi) {
				if (m_position[vi->label()] <= best_pos) {
					best_pos = m_position[vi->label()];
					deep = vi->label();
				}
			}
			scope_union[deep] |= sc;
		}
		// propagate scope unions up the tree (child contributes to its parent)
		for (size_t oi = 0; oi < m_order.size(); ++oi) {
			vindex v = m_order[oi];
			vindex p = m_parents[v];
			if (p != (vindex) -1)
				scope_union[p] |= scope_union[v];
		}
		// context = (scopes touching subtree) intersect (strict ancestors)
		for (size_t v = 0; v < n; ++v) {
			m_cache_context[v] = scope_union[v] & ancestors[v];
		}
	}

	// Build and run the WMB heuristic on the SAME order + pseudo tree so its
	// bucket structure matches the AND/OR search tree.
	m_heuristic = wmb(m_gmo);
	std::ostringstream oss;
	oss << "iBound=" << m_ibound << ",Order=MinFill,OrderIter=" << m_order_iter
		<< ",Iter=" << m_num_iter << ",Task="
		<< (m_task == Task::MAP ? "MAP" : "MMAP") << ",Debug=0";
	m_heuristic.set_properties(oss.str());
	m_heuristic.set_var_types(m_var_types);
	m_heuristic.set_query(m_query);
	m_heuristic.set_order(m_order);
	m_heuristic.set_pseudo_tree(m_parents);
	m_heuristic.run();
	m_ub = m_heuristic.ub(); // global upper bound (log space)

	// Absolute-bound offset per variable. wmb::get_heuristic returns bounds
	// relative to a per-bucket normalization the forward pass folds into logZ; the
	// absolute completion bound below x is get_heuristic(x) scaled by exp of the
	// normalization stripped within x's subtree. m_subtree_norm[x] sums those
	// log-constants over x and all its pseudo-tree descendants. Computed leaves-to-
	// root (elimination order visits descendants before their parent). This makes
	// the per-node heuristic an absolute cost bound (matching AOBF).
	m_subtree_norm.assign(n, 0.0);
	for (size_t oi = 0; oi < m_order.size(); ++oi) {
		vindex x = m_order[oi];
		m_subtree_norm[x] += m_heuristic.bucket_norm(x);
		vindex p = m_parents[x];
		if (p != (vindex) -1)
			m_subtree_norm[p] += m_subtree_norm[x];
	}

	std::cout << "[AOBB] + heuristic UB     : " << m_ub << " ("
			<< std::exp(m_ub) << ")" << std::endl;
	std::cout << "[AOBB] + exact inference  : "
			<< (m_ibound >= wstar ? "Yes" : "No") << std::endl;
	std::cout << "[AOBB] + init time        : "
			<< (timeSystem() - m_start_time) << " seconds" << std::endl;

	// Incumbent: nothing found yet (cost space => +infinity is "no solution").
	m_incumbent_cost = std::numeric_limits<double>::infinity();
	m_best_config.assign(n, 0);
	m_num_expansions = 0;
	m_num_cache_hits = 0;
	m_proved_optimal = false;
	m_found_solution = false;

	// OR value cache (one table per variable).
	m_cache.clear();
	if (m_caching)
		m_cache.resize(n);
}

// Induced width of the SUM variables after conditioning out (removing) the MAP
// variables, under the constrained elimination order (SUM variables come first).
// Conditioning fixes the MAP variables, so they are deleted from the graph: a
// fixed variable induces no fill among its neighbors and contributes 0 to width.
// This bounds the cost of exactly summing out a conditioned SUM subtree via
// solve_subproblem (exponential in this width, otherwise unguarded).
//
// NOTE: we do NOT call graphical_model::induced_width on a SUM-only order over the
// full mrf(): variables absent from the passed order default to elimination
// position 0, so MAP neighbors would be treated as already-eliminated and the
// width UNDER-counted -- the unsafe direction (it could admit an intractable exact
// eval). Here MAP variables are removed from the adjacency entirely.
size_t aobb::conditioned_sum_width() const {
	size_t nv = m_gmo.nvar();
	// SUM-only adjacency keyed by variable label (MAP vars dropped).
	std::vector<std::set<size_t> > adj(nv);
	std::vector<variable_set> mb = m_gmo.mrf();
	for (size_t v = 0; v < nv; ++v) {
		if (m_var_types[v]) continue; // MAP var: not in the conditioned graph
		for (variable_set::const_iterator it = mb[v].begin(); it != mb[v].end(); ++it) {
			vindex u = it->label();
			if (u != v && !m_var_types[u]) adj[v].insert(u); // keep SUM neighbors only
		}
	}
	// Min-fill-style elimination over the SUM variables in the constrained order
	// (SUM variables precede all MAP variables in m_order). Count, for each SUM
	// variable, its not-yet-eliminated SUM neighbors, then connect them (fill-in).
	size_t width = 0;
	std::vector<char> eliminated(nv, 0);
	for (size_t oi = 0; oi < m_order.size(); ++oi) {
		vindex x = m_order[oi];
		if (m_var_types[x]) continue; // skip MAP vars entirely
		// Neighbors of x not yet eliminated.
		std::vector<size_t> nb;
		for (std::set<size_t>::const_iterator it = adj[x].begin();
				it != adj[x].end(); ++it) {
			if (!eliminated[*it]) nb.push_back(*it);
		}
		width = std::max(width, nb.size());
		// Connect the surviving neighbors (induced fill edges).
		for (size_t i = 0; i < nb.size(); ++i)
			for (size_t j = i + 1; j < nb.size(); ++j) {
				adj[nb[i]].insert(nb[j]);
				adj[nb[j]].insert(nb[i]);
			}
		eliminated[x] = 1;
	}
	return width;
}

// Run GLS+ to obtain an initial solution and seed the incumbent cost + best
// configuration for pruning (MAP and MMAP; see the header rationale). GLS+ is built
// on the same post-evidence model and query, so its best_config uses the same
// variable indexing as m_best_config. For MAP its logZ is the negative of a true
// solution cost, used directly. For MMAP its own objective is a WMB upper bound on
// probability (lower bound on cost -- the wrong direction), so the GLS+ query
// configuration is RE-EVALUATED into a valid upper-bound cost (see below).
bool aobb::seed_incumbent() {
	if (!m_ls_seed)
		return false;

	// Cap the seed budget against the overall search time limit so the seed does
	// not consume the whole budget: leave at least half of the remaining time for
	// the branch-and-bound search itself. Never exceed the user's explicit seed
	// budget. With no time limit the seed budget is used verbatim.
	double ls_time = m_ls_time_limit;
	if (m_time_limit > 0.0) {
		double remaining = m_time_limit - (timeSystem() - m_start_time);
		if (remaining <= 0.0)
			return false; // no time left even before search starts
		ls_time = std::min(ls_time, 0.5 * remaining);
	}

	std::cout << "[AOBB] + seeding incumbent with GLS+ ("
			<< ls_time << "s, "
			<< (m_ls_max_flips > 0 ? std::to_string(m_ls_max_flips) + " flips"
					: std::string("time-bounded")) << ")" << std::endl;

	gls ls(m_gmo);
	std::ostringstream oss;
	oss << "Task=" << (m_task == Task::MAP ? "MAP" : "MMAP")
		<< ",TimeLimit=" << ls_time
		<< ",Iter=" << m_ls_max_flips   // 0 => no flip budget, time governs
		<< ",Seed=" << m_seed
		<< ",iBound=" << m_ibound;      // used by the MMAP WMB estimator
	ls.set_properties(oss.str());
	ls.set_query(m_query);
	ls.run();

	if (!ls.found_solution())
		return false;

	const std::vector<index>& cfg = ls.best_config();
	const double INF = std::numeric_limits<double>::infinity();
	size_t n = m_gmo.nvar();
	double cost;

	if (m_task == Task::MAP) {
		// GLS+ log-prob is the negative of a true MAP solution cost.
		cost = -ls.logZ();
	} else {
		// MMAP: re-evaluate x_M = cfg (MAP vars set, SUM vars == -1) into a valid
		// upper-bound cost. Q(x_M) = P(x_M, e) = Σ_{x_S} Π factors.
		std::vector<size_t> asgn(cfg.begin(), cfg.end());

		// SUM subtree roots: SUM var whose pseudo-tree parent is a MAP var or a root.
		std::vector<vindex> sum_roots;
		for (vindex v = 0; v < n; ++v) {
			if (m_var_types[v]) continue; // MAP var
			vindex p = m_parents[v];
			if (p == (vindex) -1 || m_var_types[p]) sum_roots.push_back(v);
		}

		// Exact only if the conditioned-SUM problem is tractable (width <= i-bound);
		// solve_subproblem is exponential in this width and otherwise unguarded.
		size_t sum_width = conditioned_sum_width();
		bool exact_ok = (sum_width <= m_ibound);

		double c = 0.0;
		if (exact_ok) {
			// Exact -log Q(x_M): the factors partition by anchor variable. Factors
			// with all-MAP scope are anchored at MAP vars (exact via label); every
			// other factor is anchored inside a SUM subtree (exact via the logsumexp
			// aggregation of solve_subproblem). No factor is counted twice.
			for (vindex v = 0; v < n && c < INF; ++v)
				if (m_var_types[v]) c += label(v, asgn);
			for (size_t k = 0; k < sum_roots.size() && c < INF; ++k)
				c += solve_subproblem(sum_roots[k], asgn, /*caching=*/false);
			std::cout << "[AOBB] + MMAP seed eval    : exact (SUM width "
					<< sum_width << " <= i-bound " << m_ibound << ")" << std::endl;
		} else {
			// Single-configuration lower bound on probability = upper bound on cost:
			// P(x_M, x_S*, e) with x_S* = 0 (any single term <= the full sum Q(x_M)).
			// Sum -log over ALL factors (== Σ label over ALL anchors) of the complete
			// assignment; a valid, possibly loose, upper bound on the optimal cost.
			std::vector<size_t> full(cfg.begin(), cfg.end());
			for (vindex v = 0; v < n; ++v)
				if (full[v] == (size_t) -1) full[v] = 0;
			for (vindex v = 0; v < n && c < INF; ++v)
				c += label(v, full);
			std::cout << "[AOBB] + MMAP seed eval    : single-config bound (SUM width "
					<< sum_width << " > i-bound " << m_ibound << ")" << std::endl;
		}
		cost = c;
		if (cost == INF)
			return false; // zero-probability configuration: not a usable solution
	}

	if (cost >= m_incumbent_cost)
		return false; // does not improve (e.g. the incumbent was already seeded)

	m_incumbent_cost = cost;
	m_best_config.assign(cfg.begin(), cfg.end());
	m_found_solution = true; // report the seed even if the search never beats it

	std::cout << "[AOBB] + GLS+ seed value   : " << -cost
			<< " (" << std::exp(-cost) << ")" << std::endl;
	return true;
}

// Compute the per-value label and heuristic COSTS of an OR node's AND children
// and the OR node's aggregate heuristic cost (min for a MAX variable, -logsumexp
// for a SUM variable). All costs are in negative-log space. The per-value arrays
// are cached on the node so generate_children_or can build the AND children.
// Native equivalent of the ground-truth heuristicOR.
//   label_cache[v] = -log(label(x=v))                        (arc cost)
//   heur_cache[v]  = label_cache[v] + (-log(get_heuristic) - subtree_norm[x])
//                  = the AND child's total cost bound (includes the arc cost)
double aobb::heuristic_or(ao_node* orn, std::vector<size_t>& asgn) {
	vindex x = orn->var;
	size_t dom = m_gmo.var(x).states();
	const double INF = std::numeric_limits<double>::infinity();
	double offset = m_subtree_norm[x]; // absolute-bound offset (see init())
	orn->label_cache.assign(dom, 0.0);
	orn->heur_cache.assign(dom, 0.0);
	double h = orn->is_max_or ? INF : 0.0; // MAX: min accumulator; SUM: logsumexp
	// For SUM we need a stable logsumexp over -heur_cache[v]; collect then combine.
	double sum_min = INF;
	for (size_t v = 0; v < dom; ++v) {
		asgn[x] = v;
		double lbl = label(x, asgn); // arc cost (+inf if zero probability)
		double c;
		if (lbl == INF) {
			c = INF; // dead-end value
		} else {
			double hv = m_heuristic.get_heuristic(x, asgn); // linear UB (no label)
			double comp = (hv <= 0.0) ? INF : (-std::log(hv) - offset); // abs. cost
			c = lbl + comp;
		}
		orn->label_cache[v] = lbl;
		orn->heur_cache[v] = c;
		if (orn->is_max_or)
			h = std::min(h, c);
		else
			sum_min = std::min(sum_min, c);
	}
	asgn[x] = (size_t) -1;
	if (!orn->is_max_or) { // SUM: h = -logsumexp(-heur_cache) = sum_min - log(sum)
		if (sum_min == INF) {
			h = INF;
		} else {
			double s = 0.0;
			for (size_t v = 0; v < dom; ++v)
				s += std::exp(sum_min - orn->heur_cache[v]);
			h = sum_min - std::log(s);
		}
	}
	orn->heur = h;
	return h;
}

// Initial processing: record the assignment of an AND node so the current path
// assignment reflects it while its sub-tree is explored.
bool aobb::do_process(ao_node* n, std::vector<size_t>& asgn) {
	if (n->type == AO_AND)
		asgn[n->var] = n->val;
	return false;
}

// OR-node cache read: on a hit, reuse the stored value and best sub-assignment.
bool aobb::do_caching(ao_node* n, std::vector<size_t>& asgn) {
	if (!m_caching || n->type != AO_OR)
		return false;
	vindex x = n->var;
	if (m_cache_context[x].num_states() == 0)
		return false;
	size_t key = sub2ind(m_cache_context[x], asgn);
	std::unordered_map<size_t, cache_entry>::const_iterator hit = m_cache[x].find(key);
	if (hit == m_cache[x].end())
		return false;
	n->value = hit->second.first;
	n->opt_assignment = hit->second.second;
	n->solved = true;
	n->exact = true; // only exact sub-trees are ever cached
	n->num_children_pending = 0;
	++m_num_cache_hits;
	return true;
}

// True if the node cannot improve the incumbent. SUM nodes are never pruned.
// In cost space, prune when the path lower-bound cost is >= the incumbent cost.
bool aobb::can_be_pruned(ao_node* n) const {
	if (!n->is_max_or && n->type == AO_OR)
		return false; // SUM OR node
	if (n->type == AO_AND && !m_var_types[n->var])
		return false; // AND node of a SUM variable
	return path_bound(n) >= m_incumbent_cost;
}

// Prune a node whose bound cannot improve the incumbent: mark it a solved-in-
// place dead end whose value (+inf cost) is NOT the true optimum (inexact).
bool aobb::do_pruning(ao_node* n) {
	if (!can_be_pruned(n))
		return false;
	n->value = std::numeric_limits<double>::infinity(); // dead end (cost)
	n->solved = true;
	n->exact = false;
	n->pruned = true;
	n->num_children_pending = 0;
	n->opt_assignment.clear();
	if (n->type == AO_AND)
		n->opt_assignment.push_back(std::make_pair((size_t) n->var, n->val));
	return true;
}

// Generate the AND children of an OR node (one per feasible domain value), using
// the per-value cost arrays computed by heuristic_or. For MAX variables the
// children are ordered by INCREASING cost (lowest cost / best first), which
// improves pruning and anytime behavior. Returns true if there are no children.
bool aobb::generate_children_or(ao_node* n, std::vector<ao_node*>& chi,
		std::vector<size_t>& asgn) {
	vindex x = n->var;
	size_t dom = m_gmo.var(x).states();
	const double INF = std::numeric_limits<double>::infinity();
	for (size_t v = 0; v < dom; ++v) {
		// Skip infinite-cost values for MAX vars (dead ends: an infinite cost
		// lower bound cannot yield a finite-probability solution).
		if (n->is_max_or && n->heur_cache[v] == INF)
			continue;
		ao_node* andc = new ao_node();
		andc->type = AO_AND;
		andc->var = x;
		andc->val = v;
		andc->label = n->label_cache[v];
		// AND-node heuristic excludes the arc (label) cost: total minus label.
		andc->heur = (n->label_cache[v] == INF) ? INF
				: (n->heur_cache[v] - n->label_cache[v]); // == completion cost
		andc->is_max_or = n->is_max_or;
		andc->parent = n;
		chi.push_back(andc);
	}
	if (chi.empty()) {
		// Dead end: no feasible value. Solved in place with +inf cost.
		n->value = INF;
		n->solved = true;
		n->num_children_pending = 0;
		n->opt_assignment.clear();
		return true;
	}
	// Order by increasing cost for MAX variables (lowest cost / best first).
	if (n->is_max_or)
		std::stable_sort(chi.begin(), chi.end(),
				[](const ao_node* a, const ao_node* b) {
					return (a->label + a->heur) < (b->label + b->heur);
				});
	n->children = chi;
	n->num_children_pending = chi.size();
	return false;
}

// Generate the OR children of an AND node (one per pseudo-tree child variable),
// seeding each child's heuristic cost from the current partial assignment. For
// MAX variables the children are ordered by INCREASING cost. Returns true if
// there are no children (leaf AND node, solved in place with value == label).
bool aobb::generate_children_and(ao_node* n, std::vector<ao_node*>& chi,
		std::vector<size_t>& asgn) {
	vindex x = n->var;
	if (m_children[x].empty()) {
		// Leaf AND node: solved with cost == its label cost.
		n->value = n->label;
		n->solved = true;
		n->num_children_pending = 0;
		n->opt_assignment.clear();
		n->opt_assignment.push_back(std::make_pair((size_t) x, n->val));
		return true;
	}
	for (size_t i = 0; i < m_children[x].size(); ++i) {
		vindex c = m_children[x][i];
		ao_node* orc = new ao_node();
		orc->type = AO_OR;
		orc->var = c;
		orc->is_max_or = m_var_types[c];
		heuristic_or(orc, asgn); // sets orc->heur and per-value caches
		orc->parent = n;
		chi.push_back(orc);
	}
	// Order by increasing cost for MAX variables (lowest cost / best first).
	if (m_var_types[x])
		std::stable_sort(chi.begin(), chi.end(),
				[](const ao_node* a, const ao_node* b) {
					return a->heur < b->heur;
				});
	n->children = chi;
	n->num_children_pending = chi.size();
	return false;
}

// Expand a node: generate its children and push them onto the DFS stack. Returns
// true if the node was solved in place (no children). Children are pushed so the
// first (lowest-cost, for MAX vars) is explored first (last pushed).
bool aobb::do_expand(ao_node* n, std::stack<ao_node*>& stk,
		std::vector<size_t>& asgn) {
	std::vector<ao_node*> chi;
	bool solved_in_place = (n->type == AO_OR)
			? generate_children_or(n, chi, asgn)
			: generate_children_and(n, chi, asgn);
	if (solved_in_place)
		return true;
	// Push in reverse so children[0] (best UB) is on top and explored first.
	for (size_t i = chi.size(); i-- > 0; )
		stk.push(chi[i]);
	return false;
}

// Build the dummy super-root AND node with one OR child per pseudo-tree root and
// seed the DFS stack (super-root at the bottom, its OR children on top).
ao_node* aobb::init_search_space(std::stack<ao_node*>& stk,
		const std::vector<size_t>& asgn) {
	ao_node* super = new ao_node();
	super->type = AO_AND;
	super->var = (size_t) -1; // sentinel: not a real variable
	super->label = 0.0;       // cost of the empty product = -log(1) = 0
	super->heur = 0.0;
	super->value = 0.0;
	super->parent = NULL;
	super->expanded = true; // pre-built: never regenerate its children

	std::vector<size_t> scratch(asgn); // heuristic_or mutates asgn[x] transiently
	for (size_t r = 0; r < m_roots.size(); ++r) {
		ao_node* orr = new ao_node();
		orr->type = AO_OR;
		orr->var = m_roots[r];
		orr->is_max_or = m_var_types[m_roots[r]];
		orr->parent = super;
		heuristic_or(orr, scratch); // seed heur + per-value caches (asgn all-unset)
		super->children.push_back(orr);
	}
	super->num_children_pending = super->children.size();

	stk.push(super);
	for (size_t i = 0; i < super->children.size(); ++i)
		stk.push(super->children[i]);
	return super;
}

void aobb::run() {

	// Start the timer.
	m_start_time = timeSystem();

	// Initialize order, pseudo tree, and WMB heuristic.
	init();

	size_t n = m_gmo.nvar();
	std::vector<size_t> asgn(n, (size_t) -1); // working path assignment

	// Build the super-root and seed the DFS stack.
	std::stack<ao_node*> stk;
	ao_node* super = init_search_space(stk, asgn);

	// Seed the incumbent with a Guided Local Search (GLS+) solution (MAP only), so
	// branch-and-bound can prune from the first node and a valid solution is always
	// available (anytime). No-op for MMAP or when disabled. Lowers m_incumbent_cost.
	seed_incumbent();

	// Bottom-up value propagator: aggregates solved sub-trees, writes the OR
	// cache, reports the incumbent cost. The incumbent starts at the GLS+ seed cost
	// (or +infinity if not seeded) and improves only when the search solves a
	// complete assignment with a strictly lower cost.
	bound_propagator prop(m_gmo, m_var_types, m_caching, m_cache_context, m_cache);
	prop.set_incumbent(m_incumbent_cost);

	bool timed_out = false;
	size_t steps = 0;

	// Depth-first branch-and-bound. Each node is expanded on its first visit
	// (do_process -> do_caching -> do_pruning -> do_expand). When a node is
	// solved (leaf, pruned, cache hit, or a dead end), the propagator walks its
	// value up the tree, freeing solved sub-trees and updating the incumbent.
	while (!stk.empty()) {
		// Time-limit poll (every 1024 node visits).
		if (m_time_limit > 0.0 && (++steps & 0x3FF) == 0) {
			if (timeSystem() - m_start_time >= m_time_limit) {
				timed_out = true;
				break;
			}
		}

		ao_node* node = stk.top();
		stk.pop();

		// The pre-built super-root carries no work of its own; it is solved by the
		// propagator once its OR children finish, so it never needs expanding.
		if (node == super)
			continue;

		do_process(node, asgn);          // record AND assignment
		if (do_caching(node, asgn)) {    // OR cache hit -> solved
			prop.propagate(node, asgn);
			m_incumbent_cost = prop.incumbent();
			continue;
		}
		if (do_pruning(node)) {          // bound cannot improve -> dead end
			prop.propagate(node, asgn);
			m_incumbent_cost = prop.incumbent();
			continue;
		}
		if (do_expand(node, stk, asgn)) { // solved in place (leaf / dead end)
			prop.propagate(node, asgn);
			m_incumbent_cost = prop.incumbent();
			continue;
		}
		++m_num_expansions;
		// Expanded into children: they will be explored, then this node's value
		// is aggregated by the propagator when the last child solves.
	}

	// The propagator holds the incumbent cost and assignment (the proven optimum
	// if the search completed, else the best complete solution reached by the
	// search). If the search found its own complete solution, adopt it; otherwise
	// retain the GLS+ seed (m_best_config / m_found_solution set by seed_incumbent).
	m_incumbent_cost = prop.incumbent();
	m_proved_optimal = !timed_out;
	if (prop.found_solution()) {
		m_found_solution = true;
		const std::vector<size_t>& cfg = prop.best_config();
		for (size_t i = 0; i < n; ++i) m_best_config[i] = cfg[i];
	}

	// Release the AND/OR search tree. On a clean finish only the super-root
	// remains; on a time-out the live frontier is still reachable from the
	// super-root through the children vectors, so one recursive free reclaims
	// everything.
	free_subtree(super);

	// Report the value in log space (log-prob = -cost). No solution => -inf.
	m_logz = m_found_solution ? -m_incumbent_cost
			: -std::numeric_limits<double>::infinity();

	std::cout << "[AOBB] Finished searching in "
			<< (timeSystem() - m_start_time) << " seconds" << std::endl;
	if (m_caching)
		std::cout << "[AOBB] + cache hits       : " << m_num_cache_hits << std::endl;
	if (timed_out)
		std::cout << "[AOBB] + status           : time limit reached (suboptimal)"
				<< std::endl;
	if (!m_found_solution)
		std::cout << "[AOBB] + status           : no complete solution found within "
				"the time limit" << std::endl;
	std::cout << "[AOBB] " << (m_proved_optimal ? "Optimal value      : " : "Best value so far  : ")
			<< m_logz << " (" << std::exp(m_logz) << ")" << std::endl;
}

// Solve exactly the AND/OR sub-problem rooted at 'var', conditioned on the fixed
// ancestor values in 'asgn', using the standard AOBB depth-first driver and the
// shared bound_propagator. Returns the sub-problem's solved cost (negative-log
// space). This is the single shared code path for evaluating a conditioned
// sub-problem: SUM variables are aggregated inline by the propagator's logsumexp
// rule -- identical to how the main search and the other solvers combine SUM
// nodes. It never prunes (a fresh +inf incumbent) and does not touch the outer
// search state.
double aobb::solve_subproblem(vindex var, const std::vector<size_t>& asgn,
		bool caching) {
	// Local working copy of the path assignment; the driver mutates it in place
	// but backtracks fully, so it is restored by the time the root is solved.
	std::vector<size_t> local = asgn;

	// Root OR node for the sub-problem variable.
	ao_node* root = new ao_node();
	root->type = AO_OR;
	root->var = var;
	root->is_max_or = m_var_types[var];
	root->parent = NULL;
	heuristic_or(root, local); // seed heur + per-value caches given the ancestors

	std::stack<ao_node*> stk;
	stk.push(root);

	// A propagator scoped to this sub-search. Caching uses the shared per-variable
	// cache tables only when requested; otherwise a private empty cache is used so
	// the sub-search is self-contained.
	std::vector<std::unordered_map<size_t, cache_entry> > scratch_cache;
	std::vector<variable_set> scratch_ctx;
	std::vector<std::unordered_map<size_t, cache_entry> >* cache_ptr;
	const std::vector<variable_set>* ctx_ptr;
	if (caching && m_caching) {
		cache_ptr = &m_cache;
		ctx_ptr = &m_cache_context;
	} else {
		scratch_ctx.assign(m_gmo.nvar(), variable_set());
		cache_ptr = &scratch_cache;
		ctx_ptr = &scratch_ctx;
	}
	bound_propagator prop(m_gmo, m_var_types, (caching && m_caching),
			*ctx_ptr, *cache_ptr);

	while (!stk.empty()) {
		ao_node* node = stk.top();
		stk.pop();
		do_process(node, local);
		if (caching && m_caching && do_caching(node, local)) {
			prop.propagate(node, local);
			continue;
		}
		// No incumbent-based pruning in a sub-problem solve (incumbent is +inf).
		if (do_expand(node, stk, local)) { // solved in place (leaf / dead end)
			prop.propagate(node, local);
		}
	}

	double value = root->value; // solved cost of the sub-problem
	free_subtree(root);
	return value;
}

// Write the solution to the output stream.
void aobb::write_solution(std::ostream& out, const std::map<size_t, size_t>& evidence,
		const std::map<size_t, size_t>& old2new, const graphical_model& orig,
		const std::set<size_t>& dummies, int output_format) {

	// Status: "true" (proven optimal) or "false" (time-limit reached, the
	// reported value/assignment is the best found so far, not proven optimal).
	const char* status = m_proved_optimal ? "true" : "false";

	if (output_format == MERLIN_OUTPUT_JSON) {
		out << "{";
		out << " \"algorithm\" : \"" << algo_name() << "\", ";
		out << " \"ibound\" : " << m_ibound << ", ";
		out << " \"iterations\" : " << m_num_iter << ", ";

		if (m_task == Task::MAP) {
			out << " \"task\" : \"MAP\", ";
			out << " \"value\" : " << std::fixed
				<< std::setprecision(MERLIN_PRECISION)
				<< (m_logz + std::log(orig.get_global_const())) << ", ";
			out << " \"status\" : \"" << status << "\", ";
			out << " \"optimal\" : " << (m_proved_optimal ? "true" : "false") << ", ";
			out << " \"solution\" : [ ";
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
			out << "] ";
		} else { // MMAP
			out << " \"task\" : \"MMAP\", ";
			out << " \"value\" : " << std::fixed
				<< std::setprecision(MERLIN_PRECISION)
				<< (m_logz + std::log(orig.get_global_const())) << ", ";
			out << " \"status\" : \"" << status << "\", ";
			out << " \"optimal\" : " << (m_proved_optimal ? "true" : "false") << ", ";
			out << " \"solution\" : [ ";
			for (vindex i = 0; i < m_query.size(); ++i) {
				vindex j = m_query[i];
				out << "{";
				out << " \"variable\" : " << j << ",";
				out << " \"value\" : " << m_best_config[j];
				out << "}";
				if (i != m_query.size() - 1) out << ", ";
			}
			out << "] ";
		}
		out << "}";
	} else if (output_format == MERLIN_OUTPUT_UAI) {
		if (m_task == Task::MAP) {
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
		} else { // MMAP
			out << "MMAP" << std::endl;
			out << m_query.size();
			for (vindex i = 0; i < m_query.size(); ++i) {
				vindex j = m_query[i];
				out << " " << m_best_config[j];
			}
			out << std::endl;
		}
	} else {
		std::string err_msg("Unknown output format.");
		throw std::runtime_error(err_msg);
	}
}

} // namespace merlin
