/*
 * bound_propagator.cpp
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

/// \file bound_propagator.cpp
/// \brief Bottom-up value propagation for AND/OR branch-and-bound search
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#include "bound_propagator.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace merlin {

// Aggregate a fully-solved node's children into its value (a COST) and best
// sub-assignment (opt_assignment), write the OR cache if the sub-tree is exact,
// then free the children. AND/OR aggregation rule in negative-log cost space:
//   AND: value = label_cost + sum(child costs); opt = {(var,val)} ++ child opts
//   OR (MAX var): value = min child cost; opt = argmin child.opt
//   OR (SUM var): value = -logsumexp(-child cost); opt = {} (var marginalized out)
// 'asgn' must hold this node's cache context (ancestor AND-variable values;
// descendant AND-variable values already cleared as the propagator climbed).
void bound_propagator::aggregate(ao_node* node, std::vector<size_t>& asgn) {

	if (node->children.empty()) {
		// Solved in place (leaf/pruned/cache-hit): value + opt_assignment already
		// set by the search when it created/solved the node.
		node->solved = true;
		return;
	}

	// A node is exact only if every child sub-tree was solved exactly (no
	// incumbent-based pruning anywhere inside it).
	bool all_exact = true;
	for (size_t i = 0; i < node->children.size(); ++i)
		if (!node->children[i]->exact) { all_exact = false; break; }
	node->exact = all_exact;

	if (node->type == AO_AND) {
		// AND: cost = label cost + sum of children costs.
		double val = node->label;
		std::vector<std::pair<vindex, size_t> > sub;
		// Skip the dummy super-root (var == -1); it fixes no real variable.
		if (node->var != (vindex) -1)
			sub.push_back(std::make_pair((vindex) node->var, node->val));
		for (size_t i = 0; i < node->children.size(); ++i) {
			ao_node* c = node->children[i];
			val += c->value;
			sub.insert(sub.end(), c->opt_assignment.begin(), c->opt_assignment.end());
		}
		node->value = val;
		node->opt_assignment.swap(sub);
	} else if (node->is_max_or) { // OR over a MAX var: cost = min child cost
		double val = std::numeric_limits<double>::infinity();
		int arg = -1;
		for (size_t i = 0; i < node->children.size(); ++i) {
			ao_node* c = node->children[i];
			if (arg < 0 || c->value < val) { val = c->value; arg = (int) i; }
		}
		node->value = val;
		if (arg >= 0)
			node->opt_assignment = node->children[arg]->opt_assignment;
		else
			node->opt_assignment.clear();
	} else { // OR over a SUM var: cost = -logsumexp(-child cost) (marginalize)
		// Numerically stable: -log( sum_i exp(-c_i) ) = m - log( sum_i exp(m - c_i) )
		// where m = min_i c_i (the cheapest child dominates the sum).
		double mn = std::numeric_limits<double>::infinity();
		for (size_t i = 0; i < node->children.size(); ++i)
			mn = std::min(mn, node->children[i]->value);
		double acc = 0.0;
		if (mn < std::numeric_limits<double>::infinity()) {
			for (size_t i = 0; i < node->children.size(); ++i)
				acc += std::exp(mn - node->children[i]->value);
			node->value = mn - std::log(acc);
		} else {
			node->value = std::numeric_limits<double>::infinity(); // all dead ends
		}
		// SUM node marginalizes the variable out; no MAX descendants exist under a
		// constrained pseudo tree, so an empty sub-assignment is correct.
		node->opt_assignment.clear();
	}

	// OR caching: store the sub-tree cost and its best sub-assignment keyed by the
	// context configuration -- ONLY for OR nodes whose sub-tree was solved exactly
	// (no incumbent-based pruning inside it). A pruned sub-tree's cost can be
	// overstated, so caching it would be unsound.
	if (node->type == AO_OR && m_caching && node->exact
			&& m_cache_context[node->var].num_states() > 0) {
		size_t key = sub2ind(m_cache_context[node->var], asgn);
		m_cache[node->var][key] =
				std::make_pair(node->value, node->opt_assignment);
		++m_num_cache_writes;
	}
	node->solved = true;

	// Free child nodes (their values / sub-assignments are aggregated).
	for (size_t i = 0; i < node->children.size(); ++i)
		delete node->children[i];
	node->children.clear();
}

void bound_propagator::propagate(ao_node* n, std::vector<size_t>& asgn) {

	// Climb from the just-solved node toward the root. At each step notify the
	// parent (one fewer pending child); when a parent has no pending children it
	// is aggregated, marked solved, its children freed, and we continue up. We
	// stop at the first parent that still has unsolved children.
	ao_node* node = n;
	while (true) {

		// As we ascend past a solved AND node, clear its variable from the path
		// assignment (mirrors the DFS backtrack): its descendants are done, and
		// an ancestor OR node's cache context must not see it.
		if (node->type == AO_AND)
			asgn[node->var] = (size_t) -1;

		ao_node* par = node->parent;
		if (par == NULL) {
			// Super-root solved: the optimum over the explored space is proven.
			// Branch-and-bound prunes any branch whose cost bound is >= the current
			// incumbent, so when the incumbent already equals the optimum the
			// search short-circuits and node->value (cost) can be larger. Adopt
			// node->value only if it strictly improves (lowers) the incumbent cost.
			if (node->value < m_incumbent) {
				m_incumbent = node->value;
				const std::vector<std::pair<vindex, size_t> >& sub =
						node->opt_assignment;
				for (size_t i = 0; i < sub.size(); ++i)
					m_best_config[sub[i].first] = sub[i].second;
			}
			m_found_solution = true;
			return;
		}

		if (par->num_children_pending > 0)
			par->num_children_pending--;

		if (par->num_children_pending > 0) {
			// Parent still has unsolved children: stop propagating for now.
			return;
		}

		// Parent is now fully solved: aggregate it and keep climbing.
		aggregate(par, asgn);
		node = par;
	}
}

} // namespace merlin
