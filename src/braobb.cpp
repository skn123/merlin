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

#include <queue>
#include <map>
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
	ao_node* super = new ao_node();
	super->type = AO_AND;
	super->label = 1.0;
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
	super->expanded = true;

	// Per-node best sub-assignment achieving node->value (side-map, as in aobb).
	std::map<ao_node*, std::vector<std::pair<vindex, size_t> > > best_sub;

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

	// Seed an initial incumbent for MAP (see aobb).
	if (m_task == Task::MAP) {
		std::vector<size_t> empty(n, (size_t) -1);
		update_incumbent(empty);
	}

	bool timed_out = false;
	size_t steps = 0;
	double last_incumbent_time = m_start_time;

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
			// ---- First visit: expand the node (logic identical to aobb). ----
			node->expanded = true;

			if (node->type == AO_OR) {
				vindex x = node->var;
				size_t dom = m_gmo.var(x).states();

				// OR-cache read.
				bool cache_hit = false;
				if (m_caching && m_cache_context[x].num_states() > 0) {
					size_t key = sub2ind(m_cache_context[x], asgn);
					std::unordered_map<size_t, cache_entry>::const_iterator hit =
							m_cache[x].find(key);
					if (hit != m_cache[x].end()) {
						node->value = hit->second.first;
						node->solved = true;
						node->exact = true;
						node->num_children_pending = 0;
						best_sub[node] = hit->second.second;
						++m_num_cache_hits;
						cache_hit = true;
					}
				}

				if (!cache_hit) {
					node->heur = 0.0;
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

					if (path_bound(node) <= m_lb_linear) {
						node->value = 0.0;
						node->solved = true;
						node->exact = false;
						node->num_children_pending = 0;
						best_sub[node].clear();
					} else {
						// Generate AND children (one per domain value) on the same
						// stack: an OR node is a single subproblem (no fork).
						for (size_t v = 0; v < dom; ++v) {
							ao_node* andc = new ao_node();
							andc->type = AO_AND;
							andc->var = x;
							andc->val = v;
							andc->label = child_label[v];
							andc->heur = (child_label[v] == 0.0) ? 0.0
									: (child_heur[v] / child_label[v]);
							andc->parent = node;
							node->children.push_back(andc);
						}
						node->num_children_pending = node->children.size();
						for (size_t i = 0; i < node->children.size(); ++i)
							st->push(node->children[i]);
					}
				}
			} else { // AO_AND
				vindex x = node->var;
				asgn[x] = node->val;

				if (path_bound(node) <= m_lb_linear) {
					node->value = 0.0;
					node->solved = true;
					node->exact = false;
					node->num_children_pending = 0;
					best_sub[node].clear();
					best_sub[node].push_back(std::make_pair(x, node->val));
				} else if (m_children[x].empty()) {
					node->value = node->label;
					node->solved = true;
					node->num_children_pending = 0;
					best_sub[node].clear();
					best_sub[node].push_back(std::make_pair(x, node->val));
					if (m_task == Task::MAP && m_time_limit > 0.0) {
						double now = timeSystem();
						if (now - last_incumbent_time >= 0.25) {
							update_incumbent(asgn);
							last_incumbent_time = now;
						}
					}
				} else {
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

					if (node->children.size() == 1) {
						// No decomposition: stay on the same subproblem stack.
						st->push(node->children[0]);
					} else {
						// Decomposition: fork a new subproblem stack per child so
						// the scheduler rotates breadth-first across the sibling
						// subproblems (BRAOBB's anytime edge). The children are
						// removed from this stack; this AND node stays on `st` and
						// is aggregated once all forked children finish.
						for (size_t i = 0; i < node->children.size(); ++i) {
							stack_node* cs = new stack_node(st);
							st->open_children++;
							cs->push(node->children[i]);
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
			// node resurfaced). Aggregate their values, matching aobb exactly.
			bool all_exact = true;
			for (size_t i = 0; i < node->children.size(); ++i)
				if (!node->children[i]->exact) { all_exact = false; break; }
			node->exact = all_exact;

			if (node->type == AO_AND) {
				double val = node->label;
				for (size_t i = 0; i < node->children.size(); ++i)
					val *= node->children[i]->value;
				node->value = val;
				std::vector<std::pair<vindex, size_t> > psub;
				psub.push_back(std::make_pair(node->var, node->val));
				for (size_t i = 0; i < node->children.size(); ++i) {
					std::vector<std::pair<vindex, size_t> >& cs = best_sub[node->children[i]];
					psub.insert(psub.end(), cs.begin(), cs.end());
				}
				best_sub[node] = psub;
			} else { // OR: max (query) / sum
				double val = 0.0;
				int arg = -1;
				for (size_t i = 0; i < node->children.size(); ++i) {
					double cv = node->children[i]->value;
					if (node->is_max_or) {
						if (arg < 0 || cv > val) { val = cv; arg = (int) i; }
					} else {
						val += cv;
					}
				}
				node->value = val;
				std::vector<std::pair<vindex, size_t> > psub;
				if (node->is_max_or && arg >= 0)
					psub = best_sub[node->children[arg]];
				best_sub[node] = psub;

				// OR-cache write (exact subtrees only, matches aobb).
				if (m_caching && node->exact
						&& m_cache_context[node->var].num_states() > 0) {
					size_t key = sub2ind(m_cache_context[node->var], asgn);
					m_cache[node->var][key] = std::make_pair(node->value, best_sub[node]);
				}
			}
			node->solved = true;

			// Free children (values aggregated into this node).
			for (size_t i = 0; i < node->children.size(); ++i) {
				best_sub.erase(node->children[i]);
				delete node->children[i];
			}
			node->children.clear();
		}

		// Pop the solved node from its stack and notify its parent.
		st->pop();
		ao_node* par = node->parent;
		if (par == NULL) {
			// Super-root solved: search complete => proven optimum.
			m_lb_linear = node->value;
			std::vector<std::pair<vindex, size_t> >& sub = best_sub[node];
			for (size_t i = 0; i < sub.size(); ++i)
				best_asgn[sub[i].first] = sub[i].second;
			for (size_t i = 0; i < n; ++i) m_best_config[i] = best_asgn[i];
			m_found_solution = true;
			m_proved_optimal = true;
			best_sub.erase(node);
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
	// anytime incumbent), in log space -- same as aobb.
	m_logz = (m_lb_linear > 0.0) ? std::log(m_lb_linear)
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
