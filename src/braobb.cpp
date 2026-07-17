/*
 * braobb.cpp
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

/// \file braobb.cpp
/// \brief Breadth-Rotating AND/OR Branch-and-Bound (BRAOBB) search
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#include "braobb.h"

#include <cmath>
#include <limits>
#include <queue>
#include <sstream>

namespace merlin {

namespace {

// A per-subproblem depth-first stack in the breadth-rotating frontier. The
// stacks form a tree mirroring the AND/OR decomposition: a stack is created for
// each independent MAP sub-problem, and its parent stack cannot make progress
// until all of its child stacks are exhausted (open_children == 0).
struct stack_node {
	std::vector<ao_node*> nodes;   // the DFS stack (top == back)
	stack_node* parent;            // parent subproblem stack (NULL for the root)
	size_t open_children;          // number of child stacks not yet exhausted

	stack_node(stack_node* p) : parent(p), open_children(0) {}
	bool empty() const { return nodes.empty(); }
	void push(ao_node* n) { nodes.push_back(n); }
	ao_node* top() const { return nodes.back(); }
	void pop() { nodes.pop_back(); }
};

} // anonymous namespace

void braobb::run() {

	// Start the timer and initialize order / pseudo tree / WMB heuristic /
	// caches (reused verbatim from aobb).
	m_start_time = timeSystem();
	init();

	size_t n = m_gmo.nvar();
	std::vector<size_t> asgn(n, (size_t) -1); // scratch assignment (rebuilt per node)
	std::vector<size_t> best_asgn(n, 0);

	// Dummy super-root AND node with one OR child per pseudo-tree root; solving
	// it (product over roots) yields the global optimum, exactly as in aobb.
	// Built via the shared init_search_space so the root OR nodes get their
	// per-value heuristic/label caches seeded (needed by generate_children_or).
	std::stack<ao_node*> seed_stk;
	ao_node* super = init_search_space(seed_stk, asgn);

	// Breadth-rotating frontier: a FIFO queue of subproblem stacks. Seed the
	// root stack with the super-root (bottom) then its OR children (on top), so
	// that once the children solve and pop, the super-root resurfaces and is
	// aggregated on its second visit -> proven optimum.
	std::queue<stack_node*> stacks;
	stack_node* root_stack = new stack_node(NULL);
	root_stack->push(super);
	for (size_t i = 0; i < super->children.size(); ++i)
		root_stack->push(super->children[i]);
	stacks.push(root_stack);
	size_t rotate_count = 0;

	// Seed the incumbent with a Guided Local Search (GLS+) solution (MAP only), so
	// branch-and-bound can prune from the first node and a valid solution is always
	// available (anytime). No-op for MMAP or when disabled. On success it lowers
	// m_incumbent_cost and sets m_best_config / m_found_solution; mirror the seed
	// into the local best_asgn so a later improving solution overwrites it cleanly.
	// All values are costs in negative-log space.
	if (seed_incumbent())
		best_asgn = m_best_config;

	bool timed_out = false;
	size_t steps = 0;

	// Reconstruct the partial assignment of a node's ancestors by walking the
	// parent chain (AND nodes carry var=val). Under breadth-rotation there is no
	// single DFS spine, so this per-node rebuild replaces aobb's shared scratch.
	// (defined as a lambda-free helper below via a local function object)

	while (!stacks.empty()) {

		// Time-limit poll.
		if (m_time_limit > 0.0 && (++steps & 0x3FF) == 0) {
			if (timeSystem() - m_start_time >= m_time_limit) {
				timed_out = true;
				break;
			}
		}

		// ---- Rotation scheduler: pick the next node to expand. ----
		stack_node* st = stacks.front();
		if (st->open_children > 0) {
			// Subproblem split and children still running: rotate to the back.
			stacks.pop();
			stacks.push(st);
			rotate_count = 0;
			continue;
		}
		if (st->empty()) {
			// Subproblem exhausted: notify parent and drop it.
			stacks.pop();
			if (st->parent && st->parent->open_children > 0)
				st->parent->open_children--;
			delete st;
			rotate_count = 0;
			continue;
		}
		if (m_rotate_limit > 0 && rotate_count == m_rotate_limit) {
			// Worked on this subproblem long enough: rotate to the back.
			stacks.pop();
			stacks.push(st);
			rotate_count = 0;
			continue;
		}
		++rotate_count;
		// Peek (do NOT pop yet): a node is expanded on first visit and
		// aggregated + popped on the second visit, once its children are solved
		// -- exactly mirroring aobb's proven two-phase model, distributed over
		// the rotating stacks.
		ao_node* node = st->top();

		// Rebuild the assignment of node's ancestors (var=val for AND ancestors).
		// Skip the dummy super-root (parent == NULL): it is an AND node with a
		// default var=0/val=0 that would otherwise clobber the real assignment
		// of variable 0.
		std::fill(asgn.begin(), asgn.end(), (size_t) -1);
		for (ao_node* a = node->parent; a != NULL; a = a->parent) {
			if (a->type == AO_AND && a->parent != NULL)
				asgn[a->var] = a->val;
		}

		if (!node->expanded) {
			// ---- First visit: expand the node using the shared aobb helpers. ----
			node->expanded = true;

			// do_process records the AND assignment (OR nodes: no-op).
			do_process(node, asgn);

			// OR-cache read (solved-in-place on a hit).
			if (do_caching(node, asgn)) {
				// solved; fall through to aggregation/pop below.
			} else if (do_pruning(node)) {
				// bound cannot improve: solved-in-place dead end.
			} else if (node->type == AO_OR) {
				// Generate AND children (one per feasible value) on the SAME stack:
				// an OR node is a single subproblem (no fork).
				std::vector<ao_node*> chi;
				if (!generate_children_or(node, chi, asgn)) {
					// children[0] is the best-UB child; push reversed so it is on
					// top and explored first.
					for (size_t i = chi.size(); i-- > 0; )
						st->push(chi[i]);
				}
			} else { // AO_AND
				std::vector<ao_node*> chi;
				if (!generate_children_and(node, chi, asgn)) {
					if (chi.size() == 1) {
						// No decomposition: stay on the same subproblem stack.
						st->push(chi[0]);
					} else {
						// Decomposition: fork a new subproblem stack per child so
						// the scheduler rotates breadth-first across the sibling
						// subproblems (BRAOBB's anytime edge). The children are
						// removed from this stack; this AND node stays on `st` and
						// is aggregated once all forked children finish.
						for (size_t i = 0; i < chi.size(); ++i) {
							stack_node* cs = new stack_node(st);
							st->open_children++;
							cs->push(chi[i]);
							stacks.push(cs);
						}
					}
				}
			}

			// If the node expanded into children (not solved in place), leave it
			// on the stack; it will be revisited and aggregated once the children
			// solve. Only fall through to pop+aggregate if it solved immediately.
			if (!node->solved)
				continue;
		}

		// ---- Second visit (or solved-in-place): aggregate this node, pop it. ----
		if (!node->solved) {
			// All children have been solved (they popped themselves before this
			// node resurfaced). Aggregate their values into this node. The best
			// sub-assignment is stored on the node (opt_assignment); children are
			// freed here since braobb pops a parent only after its children solve.
			bool all_exact = true;
			for (size_t i = 0; i < node->children.size(); ++i)
				if (!node->children[i]->exact) { all_exact = false; break; }
			node->exact = all_exact;

			const double INF = std::numeric_limits<double>::infinity();
			if (node->type == AO_AND) {
				// AND: cost = label cost + sum of children costs.
				double val = node->label;
				std::vector<std::pair<vindex, size_t> > psub;
				// Skip the dummy super-root (var == -1); it fixes no real variable.
				if (node->var != (vindex) -1)
					psub.push_back(std::make_pair((vindex) node->var, node->val));
				for (size_t i = 0; i < node->children.size(); ++i) {
					ao_node* c = node->children[i];
					val += c->value;
					psub.insert(psub.end(), c->opt_assignment.begin(),
							c->opt_assignment.end());
				}
				node->value = val;
				node->opt_assignment.swap(psub);
			} else if (node->is_max_or) { // OR over MAX var: cost = min child cost
				double val = INF;
				int arg = -1;
				for (size_t i = 0; i < node->children.size(); ++i) {
					double cv = node->children[i]->value;
					if (arg < 0 || cv < val) { val = cv; arg = (int) i; }
				}
				node->value = val;
				if (arg >= 0)
					node->opt_assignment = node->children[arg]->opt_assignment;
				else
					node->opt_assignment.clear();
			} else { // OR over SUM var: cost = -logsumexp(-child cost)
				double mn = INF;
				for (size_t i = 0; i < node->children.size(); ++i)
					mn = std::min(mn, node->children[i]->value);
				if (mn == INF) {
					node->value = INF;
				} else {
					double s = 0.0;
					for (size_t i = 0; i < node->children.size(); ++i)
						s += std::exp(mn - node->children[i]->value);
					node->value = mn - std::log(s);
				}
				node->opt_assignment.clear();
			}
			// OR-cache write (exact subtrees only, matches aobb).
			if (node->type == AO_OR && m_caching && node->exact
					&& m_cache_context[node->var].num_states() > 0) {
				size_t key = sub2ind(m_cache_context[node->var], asgn);
				m_cache[node->var][key] =
						std::make_pair(node->value, node->opt_assignment);
			}
			node->solved = true;

			// Free children (values aggregated into this node).
			for (size_t i = 0; i < node->children.size(); ++i)
				delete node->children[i];
			node->children.clear();
		}

		// Pop the solved node from its stack and notify its parent.
		st->pop();
		ao_node* par = node->parent;
		if (par == NULL) {
			// Super-root solved: the optimum is proven. Branch-and-bound prunes
			// any branch whose cost bound is >= the current incumbent, so when that
			// incumbent already equals the optimum the search short-circuits and
			// node->value (cost) can be larger. Adopt node->value only if it
			// strictly lowers the incumbent cost; else keep the incumbent.
			if (node->value < m_incumbent_cost) {
				m_incumbent_cost = node->value;
				const std::vector<std::pair<vindex, size_t> >& sub =
						node->opt_assignment;
				for (size_t i = 0; i < sub.size(); ++i)
					best_asgn[sub[i].first] = sub[i].second;
				for (size_t i = 0; i < n; ++i) m_best_config[i] = best_asgn[i];
			}
			m_found_solution = true;
			m_proved_optimal = true;
		} else if (par->num_children_pending > 0) {
			par->num_children_pending--;
		}
	}

	// Release the search tree. On a clean finish only the super-root remains;
	// on a time-out the live frontier is still reachable from the super-root
	// through children vectors, so one recursive free reclaims everything.
	free_subtree(super);
	// Drain any remaining stack objects (their ao_nodes were freed above).
	while (!stacks.empty()) { delete stacks.front(); stacks.pop(); }

	// Report the best value found (proven optimum if completed, else best
	// anytime incumbent), in log space (log-prob = -cost) -- same as aobb.
	m_logz = m_found_solution ? -m_incumbent_cost
			: -std::numeric_limits<double>::infinity();

	std::cout << "[BRAOBB] Finished searching in "
			<< (timeSystem() - m_start_time) << " seconds" << std::endl;
	std::cout << "[BRAOBB] + rotate limit     : " << m_rotate_limit << std::endl;
	if (m_caching)
		std::cout << "[BRAOBB] + cache hits       : " << m_num_cache_hits << std::endl;
	if (timed_out)
		std::cout << "[BRAOBB] + status           : time limit reached (suboptimal)"
				<< std::endl;
	if (!m_found_solution)
		std::cout << "[BRAOBB] + status           : no complete solution found within "
				"the time limit" << std::endl;
	std::cout << "[BRAOBB] " << (m_proved_optimal ? "Optimal value      : " : "Best value so far  : ")
			<< m_logz << " (" << std::exp(m_logz) << ")" << std::endl;
}

} // namespace merlin
