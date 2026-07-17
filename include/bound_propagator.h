/*
 * bound_propagator.h
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

/// \file bound_propagator.h
/// \brief Bottom-up value propagation for AND/OR branch-and-bound search
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_BOUND_PROPAGATOR_H_
#define IBM_MERLIN_BOUND_PROPAGATOR_H_

#include "ao_search.h"
#include "graphical_model.h"
#include "variable_set.h"

#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace merlin {

///
/// \brief Bottom-up bound propagator for AND/OR branch-and-bound search.
///
/// The propagator is the native equivalent of the DAOOPT BoundPropagator: given a
/// just-solved leaf node it walks up the AND/OR search tree, aggregating child
/// values into their parents, reconstructing the best sub-assignment on each node,
/// writing the OR value cache for exactly-solved sub-trees, and freeing the solved
/// children as it climbs. When it reaches the super-root it reports the new
/// incumbent (best complete solution found so far) if it improves.
///
/// All values are COSTS in negative-log space (\c -log(probability)): the search
/// MINIMIZES cost, which maximizes the MAP/MMAP probability. Aggregation is:
///   AND: value = label_cost + sum(child costs)
///   OR (MAX var): value = min(child costs); opt = argmin child's opt
///   OR (SUM var): value = -logsumexp(-child cost); the var is marginalized out
/// A cost of +infinity denotes a zero-probability dead end.
///
/// It intentionally holds only references to the search-owned state it needs
/// (model, caches, variable types); the search owns and creates the nodes, the
/// propagator only mutates and frees solved sub-trees.
///
class bound_propagator {
public:
	typedef graphical_model::vindex vindex;

	/// Per-variable OR cache entry: (solved sub-tree cost, best sub-assignment).
	typedef std::pair<double, std::vector<std::pair<vindex, size_t> > > cache_entry;

	///
	/// \brief Construct a propagator over the search's shared state.
	/// \param gmo           the (post-evidence) graphical model
	/// \param var_types     per-variable MAX(true)/SUM(false) flags
	/// \param caching       whether OR caching is enabled
	/// \param cache_context per-variable cache-context variable set
	/// \param cache         per-variable context-index -> cache entry tables
	/// \param cache_hits    counter to bump on nothing here (kept by search)
	///
	bound_propagator(const graphical_model& gmo,
			const std::vector<bool>& var_types, bool caching,
			const std::vector<variable_set>& cache_context,
			std::vector<std::unordered_map<size_t, cache_entry> >& cache) :
			m_gmo(gmo), m_var_types(var_types), m_caching(caching),
			m_cache_context(cache_context), m_cache(cache),
			m_incumbent(std::numeric_limits<double>::infinity()),
			m_num_cache_writes(0),
			m_best_config(gmo.nvar(), 0), m_found_solution(false) {
	}

	///
	/// \brief Propagate the value of a just-solved node up the search tree.
	///
	/// Climbs from \c n toward the root. At each parent it decrements the pending
	/// child count; when a parent has no pending children left it is aggregated
	/// (value + best sub-assignment computed from its children, OR cache written
	/// if the sub-tree is exact), marked solved, and its children freed. Climbing
	/// stops at the first parent that still has unsolved children, or at the
	/// super-root (parent == NULL), where the incumbent is updated.
	///
	/// \param n     the node whose exact \c value has just been established
	/// \param asgn  the current path assignment; used to key the OR cache and
	///              maintained (AND-variable values cleared) as the walk ascends
	///
	void propagate(ao_node* n, std::vector<size_t>& asgn);

	/// \brief Best complete-solution COST found so far (min cost; +inf if none).
	double incumbent() const { return m_incumbent; }
	/// \brief true once any complete solution has been propagated to the root.
	bool found_solution() const { return m_found_solution; }
	/// \brief The best complete assignment found so far (by post-evidence index).
	const std::vector<size_t>& best_config() const { return m_best_config; }
	/// \brief Number of OR-cache writes performed.
	size_t num_cache_writes() const { return m_num_cache_writes; }

	/// \brief Seed the incumbent cost (e.g. from a known upper bound on cost).
	void set_incumbent(double v) { m_incumbent = v; }

private:
	/// Aggregate a fully-solved node's children into its value + opt_assignment,
	/// write the OR cache if applicable, and free the children.
	void aggregate(ao_node* node, std::vector<size_t>& asgn);

	const graphical_model& m_gmo;
	const std::vector<bool>& m_var_types;
	bool m_caching;
	const std::vector<variable_set>& m_cache_context;
	std::vector<std::unordered_map<size_t, cache_entry> >& m_cache;

	double m_incumbent;                ///< incumbent cost (min cost found; +inf if none)
	size_t m_num_cache_writes;         ///< OR-cache writes
	std::vector<size_t> m_best_config; ///< incumbent assignment
	bool m_found_solution;             ///< a complete solution reached the root
};

} // namespace merlin

#endif /* IBM_MERLIN_BOUND_PROPAGATOR_H_ */
