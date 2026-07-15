/*
 * ao_search.h
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

/// \file ao_search.h
/// \brief Shared AND/OR search tree node
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_AO_SEARCH_H_
#define IBM_MERLIN_AO_SEARCH_H_

#include <cstddef>
#include <vector>

namespace merlin {

///
/// \brief Type of an AND/OR search tree node.
///
/// OR nodes represent a choice of a value for a variable; AND nodes represent
/// a specific value assignment to that variable and branch over the children
/// of the variable in the guiding pseudo tree.
///
enum ao_node_type {
	AO_OR = 0,    ///< OR node (variable choice)
	AO_AND = 1    ///< AND node (value assignment)
};

///
/// \brief A node in an AND/OR search tree.
///
/// This struct is shared infrastructure for all AND/OR search algorithms
/// (branch-and-bound, best-first, recursive best-first). It is intentionally
/// algorithm-agnostic: fields that only some algorithms need (e.g. \c priority
/// for best-first search) are present but simply left unused by algorithms that
/// do not need them.
///
struct ao_node {
	ao_node_type type;              ///< OR or AND
	size_t var;                     ///< OR: the branching variable;
	                                ///< AND: the variable whose value is fixed (== parent OR var)
	size_t val;                     ///< AND: the value assigned to \c var (unused for OR)
	double label;                   ///< AND: product of factors fully instantiated at this
	                                ///<      variable (linear space). OR: 1.0
	double heur;                    ///< Heuristic upper bound on completing the subtree below
	                                ///< this node (linear space)
	double value;                   ///< Exact value of the solved subtree rooted here (linear),
	                                ///< filled bottom-up as children are solved
	double priority;                ///< Spare priority / f-value slot for best-first searches
	bool is_max_or;                 ///< OR: true if \c var is a MAX (query) variable, so children
	                                ///< are combined by max; false => SUM (combine by sum)
	bool expanded;                  ///< Children have already been generated
	bool solved;                    ///< The exact \c value of this sub-tree is known
	bool exact;                     ///< The sub-tree was solved with NO incumbent-based pruning
	                                ///< inside it, so \c value is the true optimum (cacheable)

	ao_node* parent;                ///< Back-pointer (root has NULL)
	std::vector<ao_node*> children; ///< OR: one AND child per domain value;
	                                ///< AND: one OR child per pseudo-tree child variable
	size_t num_children_pending;    ///< Number of children not yet solved

	ao_node() :
			type(AO_OR), var(0), val(0), label(1.0), heur(0.0), value(0.0),
			priority(0.0), is_max_or(true), expanded(false), solved(false),
			exact(true), parent(NULL), num_children_pending(0) {
	}
};

} // namespace merlin

#endif /* IBM_MERLIN_AO_SEARCH_H_ */
