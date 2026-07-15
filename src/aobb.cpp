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
	return lbl;
}

// Greedily complete a partial assignment and, if the resulting complete
// assignment improves the incumbent, record it. Used for MAP anytime: any full
// assignment has an exactly computable value (product of all factors), so it is
// a valid lower bound on the optimum and always available on a time-out.
void aobb::update_incumbent(const std::vector<size_t>& partial) {
	size_t n = m_gmo.nvar();
	std::vector<size_t> a = partial;
	// Assign variables root->leaf (largest position first), so a variable's
	// ancestors are fixed before it -- greedily maximizing label * heuristic.
	for (size_t oi = m_order.size(); oi-- > 0; ) {
		vindex x = m_order[oi];
		if (x >= n || a[x] != (size_t) -1) continue; // already fixed
		size_t dom = m_gmo.var(x).states();
		size_t best_v = 0;
		double best_s = -1.0;      // best heuristic-weighted score
		size_t feasible_v = dom;   // first value with a strictly positive label
		for (size_t v = 0; v < dom; ++v) {
			a[x] = v;
			double lbl = label(x, a);
			if (lbl > 0.0 && feasible_v == dom) feasible_v = v;
			double s = (lbl == 0.0) ? 0.0 : lbl * m_heuristic.get_heuristic(x, a);
			if (s > best_s) { best_s = s; best_v = v; }
		}
		// Prefer the heuristic-best value, but if it has zero label (a hard
		// determinism conflict) fall back to any feasible value so the greedy
		// completion stays a positive-probability assignment where one exists.
		if (best_s <= 0.0 && feasible_v != dom) best_v = feasible_v;
		a[x] = best_v;
	}
	// Any variable still unset (e.g. not in the order) gets value 0.
	for (size_t v = 0; v < n; ++v)
		if (a[v] == (size_t) -1) a[v] = 0;

	double val = std::exp(m_gmo.logP(a)); // exact value of this complete assignment
	if (val > m_lb_linear) {
		m_lb_linear = val;
		m_best_config.assign(a.begin(), a.end());
		m_found_solution = true;
	}
}

// Upper bound on the best complete solution consistent with the partial path
// from the root down through node 'n'. Walk up to the root, combining the
// committed labels (AND) and the sub-tree bounds of siblings (already-solved
// value where available, heuristic otherwise) with the AND=product,
// OR=max|sum rule. This is the native equivalent of AOBB's canBePruned.
double aobb::path_bound(ao_node* n) const {
	// Bound contributed by the sub-tree rooted at 'n' itself.
	double acc = (n->type == AO_AND) ? (n->label * n->heur) : n->heur;

	ao_node* cur = n;
	ao_node* par = n->parent;
	while (par != NULL) {
		if (par->type == AO_AND) {
			// AND parent: multiply by its label and by the sibling OR nodes'
			// contributions (solved value if available, else heuristic).
			acc *= par->label;
			for (size_t i = 0; i < par->children.size(); ++i) {
				ao_node* c = par->children[i];
				if (c == cur) continue;
				acc *= (c->solved ? c->value : c->heur);
			}
		} else {
			// OR parent: combine 'acc' with the solved/heuristic values of the
			// sibling AND branches by max (query var) or sum (SUM var).
			if (par->is_max_or) {
				for (size_t i = 0; i < par->children.size(); ++i) {
					ao_node* c = par->children[i];
					if (c == cur) continue;
					double cv = (c->solved ? c->value : c->heur);
					acc = std::max(acc, cv);
				}
			} else {
				for (size_t i = 0; i < par->children.size(); ++i) {
					ao_node* c = par->children[i];
					if (c == cur) continue;
					double cv = (c->solved ? c->value : c->heur);
					acc += cv;
				}
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

	std::cout << "[AOBB] + heuristic UB     : " << m_ub << " ("
			<< std::exp(m_ub) << ")" << std::endl;
	std::cout << "[AOBB] + exact inference  : "
			<< (m_ibound >= wstar ? "Yes" : "No") << std::endl;
	std::cout << "[AOBB] + init time        : "
			<< (timeSystem() - m_start_time) << " seconds" << std::endl;

	// Incumbent: nothing found yet.
	m_lb_linear = 0.0;
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

void aobb::run() {

	// Start the timer.
	m_start_time = timeSystem();

	// Initialize order, pseudo tree, and WMB heuristic.
	init();

	size_t n = m_gmo.nvar();
	std::vector<size_t> asgn(n, (size_t) -1);        // working assignment
	std::vector<size_t> best_asgn(n, 0);             // best complete assignment (linear space)

	// A dummy super-root AND node holds one OR child per pseudo-tree root, so
	// the whole forest is solved as a single AND (product of the roots).
	ao_node* super = new ao_node();
	super->type = AO_AND;
	super->label = 1.0;
	super->heur = 1.0;
	super->value = 1.0;
	super->parent = NULL;

	for (size_t r = 0; r < m_roots.size(); ++r) {
		ao_node* orr = new ao_node();
		orr->type = AO_OR;
		orr->var = m_roots[r];
		orr->is_max_or = m_var_types[m_roots[r]];
		orr->heur = m_heuristic.get_heuristic(m_roots[r], asgn); // asgn all-unset
		orr->parent = super;
		super->children.push_back(orr);
	}
	super->num_children_pending = super->children.size();
	// The super-root is pre-built: mark it expanded so the DFS loop does not
	// re-generate its children via the generic AND logic.
	super->expanded = true;

	// Per-node best sub-assignment (variable -> value) achieving node->value.
	// Kept in a side map so ao_node stays algorithm-agnostic.
	std::map<ao_node*, std::vector<std::pair<vindex, size_t> > > best_sub;

	std::stack<ao_node*> stk;
	stk.push(super);
	// Push the pre-built root OR children so the search actually descends.
	for (size_t i = 0; i < super->children.size(); ++i)
		stk.push(super->children[i]);

	// Seed an initial incumbent by greedily completing the empty assignment
	// (MAP only: for MAP every full assignment has an exactly computable value).
	// This guarantees a valid best-so-far solution even if the search is cut off
	// almost immediately.
	if (m_task == Task::MAP) {
		std::vector<size_t> empty(n, (size_t) -1);
		update_incumbent(empty);
	}

	bool timed_out = false;
	size_t steps = 0;
	double last_incumbent_time = m_start_time; // last time the greedy incumbent ran

	while (!stk.empty()) {
		// Time-limit check. Poll the clock every 1024 node visits so the limit
		// is honored promptly without calling timeSystem() on every step.
		if (m_time_limit > 0.0 && (++steps & 0x3FF) == 0) {
			if (timeSystem() - m_start_time >= m_time_limit) {
				timed_out = true;
				break;
			}
		}

		ao_node* node = stk.top();

		if (!node->expanded) {
			node->expanded = true;

			if (node->type == AO_OR) {
				vindex x = node->var;
				size_t dom = m_gmo.var(x).states();

				// OR caching: if the sub-problem below x for the current context
				// assignment has been solved before, reuse the stored value and
				// best sub-assignment instead of re-searching.
				if (m_caching && m_cache_context[x].num_states() > 0) {
					size_t key = sub2ind(m_cache_context[x], asgn);
					std::unordered_map<size_t, cache_entry>::const_iterator hit =
							m_cache[x].find(key);
					if (hit != m_cache[x].end()) {
						node->value = hit->second.first;
						node->solved = true;
						node->num_children_pending = 0;
						best_sub[node] = hit->second.second;
						node->children.clear();
						++m_num_cache_hits;
						// fall through to aggregation/propagation (pending == 0)
						continue; // re-visit: expanded && solved => propagate
					}
				}

				// Compute per-value heuristics; prune the whole OR node if it
				// cannot possibly improve the incumbent.
				node->heur = node->is_max_or ? 0.0 : 0.0;
				std::vector<double> child_heur(dom, 0.0);
				std::vector<double> child_label(dom, 0.0);
				for (size_t v = 0; v < dom; ++v) {
					asgn[x] = v;
					double lbl = label(x, asgn);
					double h = (lbl == 0.0) ? 0.0 : m_heuristic.get_heuristic(x, asgn);
					child_label[v] = lbl;
					child_heur[v] = lbl * h;
					if (node->is_max_or)
						node->heur = std::max(node->heur, child_heur[v]);
					else
						node->heur += child_heur[v];
				}
				asgn[x] = (size_t) -1;

				// Prune-check on this OR node.
				if (path_bound(node) <= m_lb_linear) {
					// Cannot improve: mark solved with value 0 and propagate.
					// The value is NOT the true sub-tree optimum, so mark the
					// node inexact (it must not be cached, and its ancestors
					// become inexact too).
					node->value = 0.0;
					node->solved = true;
					node->exact = false;
					node->num_children_pending = 0;
					best_sub[node].clear();
					node->children.clear();
				} else {
					// Generate AND children (one per domain value).
					for (size_t v = 0; v < dom; ++v) {
						ao_node* andc = new ao_node();
						andc->type = AO_AND;
						andc->var = x;
						andc->val = v;
						andc->label = child_label[v];
						andc->heur = (child_label[v] == 0.0) ? 0.0
								: (child_heur[v] / child_label[v]); // == get_heuristic
						andc->parent = node;
						node->children.push_back(andc);
					}
					node->num_children_pending = node->children.size();
					node->value = node->is_max_or ? 0.0 : 0.0; // accumulator
					// Push children (natural order; DFS explores them in turn).
					for (size_t i = 0; i < node->children.size(); ++i)
						stk.push(node->children[i]);
				}
			} else { // AO_AND
				vindex x = node->var;
				asgn[x] = node->val;

				// Prune-check for this AND branch.
				if (path_bound(node) <= m_lb_linear) {
					node->value = 0.0;
					node->solved = true;
					node->exact = false; // pruned: value is not the true optimum
					node->num_children_pending = 0;
					best_sub[node].clear();
					best_sub[node].push_back(std::make_pair(x, node->val));
					node->children.clear();
					asgn[x] = (size_t) -1;
				} else if (m_children[x].empty()) {
					// Leaf AND node: solved with value == label.
					node->value = node->label;
					node->solved = true;
					node->num_children_pending = 0;
					best_sub[node].clear();
					best_sub[node].push_back(std::make_pair(x, node->val));
					// leave asgn[x] set until this node is popped/aggregated

					// Anytime (MAP): the current spine is a partial assignment;
					// greedily complete it to refresh the best-so-far incumbent.
					// Throttled to avoid dominating the search on large models
					// (a full greedy sweep is O(nvar) per call).
					if (m_task == Task::MAP && m_time_limit > 0.0) {
						double now = timeSystem();
						if (now - last_incumbent_time >= 0.25) {
							update_incumbent(asgn);
							last_incumbent_time = now;
						}
					}
				} else {
					// Generate OR children (one per pseudo-tree child variable).
					// Seed each child's heuristic now (given the current partial
					// assignment) so sibling bounds are valid before expansion.
					for (size_t i = 0; i < m_children[x].size(); ++i) {
						vindex c = m_children[x][i];
						ao_node* orc = new ao_node();
						orc->type = AO_OR;
						orc->var = c;
						orc->is_max_or = m_var_types[c];
						orc->heur = m_heuristic.get_heuristic(c, asgn);
						orc->parent = node;
						node->children.push_back(orc);
					}
					node->num_children_pending = node->children.size();
					node->value = node->label; // accumulator (product)
					for (size_t i = 0; i < node->children.size(); ++i)
						stk.push(node->children[i]);
				}
			}

			// If not solved immediately, continue expanding children first.
			if (node->num_children_pending > 0)
				continue;
		}

		// Node is fully expanded and all children are solved (or it solved
		// immediately). Aggregate children into this node's value, propagate to
		// the parent, then pop and free children.
		stk.pop();

		if (!node->children.empty()) {
			// A node is exact only if every child sub-tree was solved exactly
			// (no incumbent-based pruning anywhere inside it).
			bool all_exact = true;
			for (size_t i = 0; i < node->children.size(); ++i)
				if (!node->children[i]->exact) { all_exact = false; break; }
			node->exact = all_exact;

			if (node->type == AO_AND) {
				double val = node->label;
				std::vector<std::pair<vindex, size_t> > sub;
				sub.push_back(std::make_pair(node->var, node->val));
				for (size_t i = 0; i < node->children.size(); ++i) {
					ao_node* c = node->children[i];
					val *= c->value;
					std::vector<std::pair<vindex, size_t> >& csub = best_sub[c];
					sub.insert(sub.end(), csub.begin(), csub.end());
					best_sub.erase(c);
				}
				node->value = val;
				best_sub[node] = sub;
			} else { // AO_OR: combine AND children by max / sum
				double val = node->is_max_or ? 0.0 : 0.0;
				int arg = -1;
				for (size_t i = 0; i < node->children.size(); ++i) {
					ao_node* c = node->children[i];
					if (node->is_max_or) {
						if (arg < 0 || c->value > val) { val = c->value; arg = (int) i; }
					} else {
						val += c->value;
					}
				}
				node->value = val;
				std::vector<std::pair<vindex, size_t> > sub;
				if (node->is_max_or) {
					if (arg >= 0) sub = best_sub[node->children[arg]];
				} else {
					// SUM node: value marginalizes the variable out; any child's
					// sub-assignment for this var would be spurious, so record
					// none for the SUM variable itself but keep descendant MAX
					// choices from the (arbitrary) first child that carries them.
					// For MMAP correctness only query (MAX) variables matter in
					// the output, and SUM variables have no descendants that are
					// MAX (constrained pseudo tree), so an empty sub is correct.
					sub.clear();
				}
				best_sub[node] = sub;
				for (size_t i = 0; i < node->children.size(); ++i)
					best_sub.erase(node->children[i]);

				// OR caching: store the sub-tree value and its best sub-assignment,
				// keyed by the context configuration -- but ONLY if the sub-tree
				// was solved exactly (no incumbent-based pruning inside it). A
				// pruned sub-tree's value can be understated, so caching it would
				// be unsound; such nodes are left out of the cache.
				if (m_caching && node->exact
						&& m_cache_context[node->var].num_states() > 0) {
					size_t key = sub2ind(m_cache_context[node->var], asgn);
					m_cache[node->var][key] = std::make_pair(node->value, best_sub[node]);
				}
			}
		}
		node->solved = true; // value is now final (aggregated or solved in place)

		// Free child nodes (their values/sub-assignments are aggregated).
		for (size_t i = 0; i < node->children.size(); ++i)
			delete node->children[i];
		node->children.clear();

		// Clear the assignment made by an AND node as we backtrack out of it.
		if (node->type == AO_AND)
			asgn[node->var] = (size_t) -1;

		// Propagate to the parent.
		if (node->parent != NULL) {
			ao_node* par = node->parent;
			if (par->num_children_pending > 0)
				par->num_children_pending--;
		} else {
			// This is the super-root: the search has completed, so the optimum is
			// proven. node->value is the best value found by the search itself;
			// but branch-and-bound prunes any branch whose bound is <= the
			// current incumbent (a value already seeded/found by the greedy
			// update_incumbent), so when that incumbent already equals the
			// optimum the search short-circuits and node->value can be smaller.
			// Adopt node->value only if it strictly improves on the incumbent;
			// otherwise keep the incumbent's value and assignment.
			if (node->value > m_lb_linear) {
				m_lb_linear = node->value;
				std::vector<std::pair<vindex, size_t> >& sub = best_sub[node];
				for (size_t i = 0; i < sub.size(); ++i)
					best_asgn[sub[i].first] = sub[i].second;
				for (size_t i = 0; i < n; ++i) m_best_config[i] = best_asgn[i];
			}
			m_found_solution = true;
			m_proved_optimal = true;
			best_sub.erase(node);
			++m_num_expansions;
		}
	}

	// Release the AND/OR search tree. Every live node is reachable from the
	// super-root through the children vectors (a node is removed from its
	// parent's children only when the parent is aggregated, which frees it), so
	// a single recursive free reclaims everything -- the whole remaining forest
	// on a time-out, or just the super-root after a completed search.
	free_subtree(super);

	// m_best_config / m_lb_linear already hold the best solution found (the
	// proven optimum if the search completed, otherwise the best anytime
	// incumbent). Report the value in log space.
	m_logz = (m_lb_linear > 0.0) ? std::log(m_lb_linear)
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
