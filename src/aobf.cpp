/*
 * aobf.cpp
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

/// \file aobf.cpp
/// \brief Best-First AND/OR Search (AOBF / AO*) with weighted mini-bucket heuristics
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#include "aobf.h"
#include "bound_propagator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stack>

namespace merlin {

namespace {
const double AOBF_INF = std::numeric_limits<double>::infinity();
// Convert a linear-space (probability) value to cost space: cost = -log(p).
// A zero probability maps to +infinity (a hard dead end).
inline double to_cost(double p) {
	return (p <= 0.0) ? AOBF_INF : -std::log(p);
}
} // anonymous namespace

// Context signature string identifying a node in the context-minimal graph.
//
// For an OR node of variable x, pass val == (size_t)-1 and C = the OR context of
// x (its relevant ancestors, m_cache_context[x]); the signature instantiates
// those ancestors from the current MAP path (x itself is not in C). For an AND
// node <x,val>, pass the assigned val and C = the AND context of x
// (m_cache_context[x] plus x); the signature additionally pins x=val.
//
// Nodes with an empty context (pseudo-tree roots, and the dummy super-root with
// var==-1) get a unique "s<counter>" signature so they never merge -- distinct
// roots are independent sub-problems.
std::string aobf::context(size_t var, size_t val, const variable_set& C) {
	if (var == (size_t) -1 || C.num_states() == 0) {
		std::ostringstream oss;
		oss << "s" << m_global_search_index++;
		return oss.str();
	}
	std::ostringstream oss;
	for (variable_set::const_iterator ci = C.begin(); ci != C.end(); ++ci) {
		size_t cvar = ci->label();
		if (cvar == var) {
			// AND node's own variable (present only in the AND context).
			oss << "x" << var << "=" << val << ";";
		} else {
			// ancestor context variable: assigned on the current MAP path.
			oss << "x" << cvar << "=" << m_assignment[cvar] << ";";
		}
	}
	return oss.str();
}

// Heuristic (cost) of an OR node. For each domain value v of the node variable:
//   weight[v] = -log(label(var, asgn))                (arc cost OR -> AND)
//   cost[v]   = -log(get_heuristic(var, asgn)) + weight[v]   (includes label)
// The OR node heuristic is the min over v (MAX variable) or the sum over v
// (SUM variable). The per-value arrays are cached on the node for expand().
double aobf::heuristic_or_cost(aobf_node* n) {
	size_t x = n->var();
	size_t dom = m_gmo.var(x).states();
	bool max_var = m_var_types[x];
	double* dv = new double[dom * 2];
	double h = max_var ? AOBF_INF : 0.0;
	size_t saved = m_assignment[x];
	// Absolute completion-bound offset for x's subtree (see run()): the normalized
	// get_heuristic must be scaled back to an absolute bound by subtracting the
	// stripped log-normalization of x's subtree (in cost space).
	double offset = m_subtree_norm[x];
	for (size_t v = 0; v < dom; ++v) {
		m_assignment[x] = v;
		double w = label(x, m_assignment);            // arc weight (COST; aobb::label
		                                              // returns -log(label) directly)
		double hv = m_heuristic.get_heuristic(x, m_assignment); // linear normalized UB
		// Absolute completion cost below x, plus the OR->AND arc weight.
		double comp = (hv <= 0.0) ? AOBF_INF : (to_cost(hv) - offset);
		double c = (w == AOBF_INF) ? AOBF_INF : (comp + w);
		dv[2 * v] = c;                                // cost (includes weight)
		dv[2 * v + 1] = w;                            // weight
		if (max_var)
			h = std::min(h, c);
		else
			h += c;
	}
	m_assignment[x] = saved;
	n->set_heur(h);
	n->set_heur_cache(dv);
	return h;
}

// Revise a node's value from its children. Returns true if the value or the
// solved status changed (so the change must propagate to parents).
bool aobf::revise(aobf_node* node) {
	bool change = true;
	if (node->type() == AO_AND) {
		if (node->is_terminal()) {
			// Terminal AND node: cost 0 (== -log(1)), solved.
			node->set_value(0.0);
			node->set_solved(true);
			node->set_fringe(false);
			change = true;
		} else {
			double old_value = node->value();
			bool solved = true;
			double qval = 0.0; // AND = sum of children costs
			for (std::list<aobf_node*>::const_iterator li = node->children().begin();
					li != node->children().end(); ++li) {
				aobf_node* ch = *li;
				solved &= ch->is_solved();
				qval += ch->value();
			}
			node->set_value(qval);
			if (solved) {
				node->set_solved(true);
				node->set_fringe(false);
			}
			change = (solved || (qval != old_value));
		}
	} else { // OR node: min over children of weight(val) + child value
		double old_value = node->value();
		double qval = AOBF_INF;
		// Seed best to the first child so a marked child always exists, even when
		// every branch is a +inf-cost (zero-probability) dead end.
		aobf_node* best = node->children().empty() ? NULL : node->children().front();
		for (std::list<aobf_node*>::const_iterator li = node->children().begin();
				li != node->children().end(); ++li) {
			aobf_node* ch = *li;
			size_t v = ch->val();
			double q = node->weight(v) + ch->value(); // could be +inf
			if (q < qval) {
				qval = q;
				best = ch;
			}
		}
		assert(best != NULL); // OR nodes always have a marked child
		node->set_value(qval);
		node->set_best_child(best);
		bool solved = best->is_solved();
		if (solved) {
			node->set_solved(true);
			node->set_fringe(false);
		}
		change = (solved || (qval != old_value));
	}
	return change;
}

// Expand a search node by generating its successors in the context-minimal graph.
// Returns true if the node has no children.
bool aobf::expand(aobf_node* node) {
	bool no_children = true;
	if (node->type() == AO_AND) { // AND node -> OR children (pseudo-tree children)
		size_t var = node->var();
		int depth = node->depth();
		const std::vector<vindex>& kids = m_children[var];
		for (size_t i = 0; i < kids.size(); ++i) {
			size_t vchild = kids[i];
			aobf_or_node* c = new aobf_or_node(vchild, depth + 1);
			c->add_parent(node);   // OR nodes have a single (current) parent
			node->add_child(c);

			if (m_var_types[vchild]) { // MAP child
				double h = heuristic_or_cost(c);
				c->set_heur(h);
				c->set_value(m_epsilon * h);
			} else { // conditioned SUM sub-problem rooted at this OR node
				double qcost = solve_sum(vchild); // exact cost (-log partition)
				c->set_heur(0.0);
				c->set_value(qcost);
				c->set_terminal(true);
				c->set_solved(true);
				c->set_fringe(false);
				c->set_expanded(true);
				++m_num_sum_evals;
			}

			// Register the OR child. OR nodes are never merged (each has a single
			// parent), so use a unique signature to avoid map collisions.
			std::string str = context((size_t) -1, (size_t) -1, variable_set());
			aobf_state state(AO_OR, str);
			m_graph->add(state, c);
			no_children = false;
		}
		node->set_expanded(true);
		node->set_fringe(false);
		node->set_terminal(kids.empty());
		m_graph->inc_expanded(AO_AND);
	} else { // OR node -> AND children (domain values), merged via the graph
		size_t var = node->var();
		int depth = node->depth();
		size_t dom = m_gmo.var(var).states();
		double* hc = node->heur_cache();
		assert(hc != NULL);
		for (size_t val = 0; val < dom; ++val) {
			// Context of the AND child: the AND context of var (ancestors + var).
			std::string str = context(var, val, m_and_context[var]);
			aobf_state state(AO_AND, str);
			if (m_graph->find(state)) {
				// Merge: reuse the existing AND node (adds a parent).
				aobf_node* c = m_graph->get(state);
				node->add_child(c);
				c->add_parent(node);
				++m_num_cache_hits;
			} else {
				aobf_and_node* c = new aobf_and_node(var, val, depth + 1);
				node->add_child(c);
				c->add_parent(node);
				double h = hc[2 * val];     // cost (includes weight)
				double w = hc[2 * val + 1]; // weight
				// AND-node heur excludes the arc weight: h - w. When both are +inf
				// (a zero-probability value) this is inf - inf = NaN; treat the
				// child as a dead end with +inf completion cost instead.
				double comp = (h == AOBF_INF) ? AOBF_INF : (h - w);
				c->set_heur(comp);
				c->set_value(m_epsilon * comp);
				m_graph->add(state, c);
			}
			no_children = false;
		}
		node->set_expanded(true);
		node->set_fringe(false);
		node->set_terminal(false);
		m_graph->inc_expanded(AO_OR);
	}
	return no_children;
}

// Expand a tip node and propagate revised values up the graph (Nilsson's
// algorithm). The multiset S holds the ancestor nodes to revise, ordered by
// their index (number of unrevised marked descendants), so a node is revised
// only after all of its relevant descendants have been.
void aobf::expand_and_revise(aobf_node* node) {

	struct comp_index {
		bool operator()(const aobf_node* a, const aobf_node* b) const {
			return a->index() < b->index();
		}
	};
	std::multiset<aobf_node*, comp_index> S;

	assert(node->is_fringe() && !node->is_solved());
	expand(node);

	S.insert(node);
	node->set_visited(true);

	while (!S.empty()) {
		aobf_node* e = *S.begin();
		S.erase(S.begin());
		e->set_visited(false);
		assert(e->index() == 0);

		bool change = revise(e);

		if (change) {
			if (e->type() == AO_AND) {
				// AND nodes may have multiple parents in the context-minimal graph.
				for (std::list<aobf_node*>::iterator pi = e->parents().begin();
						pi != e->parents().end(); ++pi) {
					aobf_node* p = *pi;
					aobf_node* best = p->best_child();
					bool found = (best == e); // is e the marked child of p?
					size_t index = 0;
					for (std::list<aobf_node*>::iterator ci = p->children().begin();
							ci != p->children().end(); ++ci)
						if ((*ci)->is_visited()) ++index;

					if (p->is_visited()) {
						// already in S: decrease its index in place
						std::multiset<aobf_node*, comp_index>::iterator si = S.begin();
						for (; si != S.end(); ++si)
							if (p == *si) { S.erase(si); break; }
						p->dec_index();
						S.insert(p);
					} else if (found) {
						// new parent reached through its marked connector
						p->set_index(index);
						p->set_visited(true);
						S.insert(p);
					}
				}
			} else {
				// OR nodes have a single parent in the context-minimal graph.
				for (std::list<aobf_node*>::iterator pi = e->parents().begin();
						pi != e->parents().end(); ++pi) {
					aobf_node* p = *pi;
					size_t index = 0;
					for (std::list<aobf_node*>::iterator ci = p->children().begin();
							ci != p->children().end(); ++ci)
						if ((*ci)->is_visited()) ++index;
					p->set_index(index);
					p->set_visited(true);
					S.insert(p);
				}
			}
		} else {
			// No change: still decrement the index of any visited parents.
			for (std::list<aobf_node*>::iterator pi = e->parents().begin();
					pi != e->parents().end(); ++pi) {
				aobf_node* p = *pi;
				if (p->is_visited()) {
					std::multiset<aobf_node*, comp_index>::iterator si = S.begin();
					for (; si != S.end(); ++si)
						if (p == *si) { S.erase(si); break; }
					p->dec_index();
					S.insert(p);
				}
			}
		}
	}

	// Recompute the best partial solution tree from the (now revised) markings.
	m_tips.clear();
	find_best_partial_tree();
	assert(m_graph->root()->is_solved() || !m_tips.empty());
}

// Follow the "best child" markings from the root to rebuild the current best
// partial solution tree: set each node's current parent, collect unexpanded
// fringe nodes into m_tips, and rebuild the current MAP path assignment.
bool aobf::find_best_partial_tree() {
	aobf_node* root = m_graph->root();
	std::fill(m_assignment.begin(), m_assignment.end(), (size_t) -1);

	std::stack<aobf_node*> s;
	s.push(root);
	while (!s.empty()) {
		aobf_node* e = s.top();
		s.pop();

		if (e->type() == AO_AND) {
			// Record the AND assignment on the current path.
			if (e->var() != (size_t) -1)
				m_assignment[e->var()] = e->val();
			const std::list<aobf_node*>& succ = e->children();
			if (!succ.empty()) { // expanded AND node: push all OR children
				for (std::list<aobf_node*>::const_iterator ci = succ.begin();
						ci != succ.end(); ++ci) {
					aobf_node* c = *ci;
					c->set_current_parent(e);
					if (!c->is_solved())
						s.push(c);
				}
			} else { // unexpanded AND node -> tip
				e->set_fringe(true);
				m_tips.push_back(e);
			}
		} else { // OR node: descend through the marked best child only
			const std::list<aobf_node*>& succ = e->children();
			if (!succ.empty()) { // expanded OR node
				aobf_node* m = e->best_child();
				assert(m != NULL);
				m->set_current_parent(e);
				s.push(m);
			} else { // unexpanded OR node -> tip
				e->set_fringe(true);
				m_tips.push_back(e);
			}
		}
	}
	return !m_tips.empty();
}

// Sort the tip nodes ascending by heuristic cost (best/cheapest first).
void aobf::arrange_tip_nodes() {
	std::sort(m_tips.begin(), m_tips.end(),
			[](const aobf_node* a, const aobf_node* b) {
				return a->heur() < b->heur();
			});
}

// Choose the next tip node to expand.
aobf_node* aobf::choose_tip_node() {
	return m_tips.empty() ? NULL : *m_tips.begin();
}

// Solve the conditioned SUM sub-problem rooted at 'var' exactly, given the
// current MAP path assignment (m_assignment). This defers to the shared
// aobb::solve_subproblem driver, so the SUM is aggregated inline by the same
// bound_propagator logsumexp rule the other MMAP solvers use -- one shared code
// path, no bespoke SUM search. Returns the sub-problem's cost (-log partition).
double aobf::solve_sum(size_t var) {
	return solve_subproblem(var, m_assignment, /*caching=*/false);
}

// The AO* graph-search loop.
int aobf::aostar() {
	size_t n = m_gmo.nvar();

	// Dummy super-root AND node (var == -1, no assignment): its OR children are
	// the pseudo-tree roots, and its value is their sum (AND aggregation). This
	// mirrors the ground truth's dummy root variable and correctly handles a
	// pseudo tree with several MAP roots. The super-root is created already
	// expanded (its OR children are pre-built and seeded as the initial tips).
	assert(!m_roots.empty());
	std::fill(m_assignment.begin(), m_assignment.end(), (size_t) -1);

	aobf_and_node* root = new aobf_and_node((size_t) -1, 0, -1);
	root->set_heur(0.0);
	root->set_value(0.0);
	root->set_terminal(false);
	root->set_fringe(false);
	root->set_expanded(true);
	// Unique context so the super-root never merges with anything.
	aobf_state rstate(AO_AND, context((size_t) -1, (size_t) -1, variable_set()));
	m_graph->set_root(root);
	m_graph->add(rstate, root);

	for (size_t r = 0; r < m_roots.size(); ++r) {
		size_t root_var = m_roots[r];
		aobf_or_node* orn = new aobf_or_node(root_var, 0);
		heuristic_or_cost(orn);       // seed per-value caches for the root variable
		orn->set_value(m_epsilon * orn->heur());
		orn->set_terminal(false);
		orn->set_fringe(true);
		orn->add_parent(root);
		root->add_child(orn);
		aobf_state ostate(AO_OR, context(root_var, (size_t) -1, variable_set()));
		m_graph->add(ostate, orn);
		m_tips.push_back(orn);
	}

	int result = 0; // 0 = success, 1 = timeout
	while (!root->is_solved()) {
		if (m_time_limit > 0.0 &&
				(timeSystem() - m_start_time) >= m_time_limit) {
			result = 1;
			break;
		}
		assert(!m_tips.empty());
		arrange_tip_nodes();
		aobf_node* tip = choose_tip_node();
		expand_and_revise(tip);
	}

	if (root->is_solved()) {
		// root->value() is the optimal cost (-log prob). Convert back to log-prob.
		m_incumbent_cost = root->value();
		m_logz = -root->value();
		// Reconstruct the MAP assignment by following markings (already rebuilt in
		// find_best_partial_tree, but that clears on unsolved tips; rebuild once).
		std::fill(m_best_config.begin(), m_best_config.end(), 0);
		std::stack<aobf_node*> s;
		s.push(root);
		while (!s.empty()) {
			aobf_node* e = s.top();
			s.pop();
			if (e->type() == AO_AND) {
				if (e->var() != (size_t) -1)
					m_best_config[e->var()] = e->val();
				for (std::list<aobf_node*>::const_iterator ci = e->children().begin();
						ci != e->children().end(); ++ci)
					s.push(*ci);
			} else {
				aobf_node* m = e->best_child();
				if (m != NULL) s.push(m);
			}
		}
		m_found_solution = true;
	}

	(void) n;
	return result;
}

void aobf::run() {

	// Start the timer and initialize order / pseudo tree / WMB heuristic (aobb).
	m_start_time = timeSystem();
	init();

	std::cout << "[AOBF] + weight (epsilon)  : " << m_epsilon << std::endl;

	// AND-node context per variable = OR context (m_cache_context, the relevant
	// ancestors) plus the variable itself. Used to hash/merge AND nodes so that
	// identical sub-problems collapse into one node (the context-minimal graph).
	// If caching was disabled m_cache_context is empty and no merging occurs
	// (the graph degenerates to a tree, still correct, just larger).
	size_t nv = m_gmo.nvar();
	m_and_context.assign(nv, variable_set());
	for (size_t v = 0; v < nv; ++v) {
		if (m_cache_context[v].num_states() > 0)
			m_and_context[v] = m_cache_context[v] | m_gmo.var(v);
	}

	// Absolute-bound offset per variable. wmb::get_heuristic returns bounds
	// relative to a per-bucket normalization that the forward pass folds into
	// logZ; the absolute completion bound below x is get_heuristic(x) scaled by
	// exp of the normalization stripped within x's subtree. m_subtree_norm[x] is
	// the sum of those log-constants over x and all its pseudo-tree descendants.
	// Computed leaves-to-root (elimination order visits descendants first).
	m_subtree_norm.assign(nv, 0.0);
	for (size_t oi = 0; oi < m_order.size(); ++oi) {
		vindex x = m_order[oi];
		m_subtree_norm[x] += m_heuristic.bucket_norm(x);
		vindex p = m_parents[x];
		if (p != (vindex) -1)
			m_subtree_norm[p] += m_subtree_norm[x];
	}

	// Search state.
	m_graph = new aobf_search_space();
	m_assignment.assign(m_gmo.nvar(), (size_t) -1);
	m_global_search_index = 0;
	m_num_sum_evals = 0;
	m_num_cache_hits = 0;

	int status = aostar();
	bool timed_out = (status != 0);
	m_proved_optimal = !timed_out;

	if (!m_found_solution) {
		m_logz = -std::numeric_limits<double>::infinity();
		m_incumbent_cost = std::numeric_limits<double>::infinity();
	}

	// Report statistics.
	std::cout << "[AOBF] Finished searching in "
			<< (timeSystem() - m_start_time) << " seconds" << std::endl;
	std::cout << "[AOBF] + OR nodes         : " << m_graph->or_nodes() << std::endl;
	std::cout << "[AOBF] + AND nodes        : " << m_graph->and_nodes() << std::endl;
	std::cout << "[AOBF] + cache hits (merge): " << m_num_cache_hits << std::endl;
	if (m_task == Task::MMAP)
		std::cout << "[AOBF] + SUM evaluations  : " << m_num_sum_evals << std::endl;
	if (timed_out)
		std::cout << "[AOBF] + status           : time limit reached (suboptimal)"
				<< std::endl;
	if (!m_found_solution)
		std::cout << "[AOBF] + status           : no complete solution found within "
				"the time limit" << std::endl;
	std::cout << "[AOBF] " << (m_proved_optimal ? "Optimal value      : " : "Best value so far  : ")
			<< m_logz << " (" << std::exp(m_logz) << ")" << std::endl;

	// Release the search graph.
	delete m_graph;
	m_graph = NULL;
}

} // namespace merlin
