/*
 * aobf.h
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

/// \file aobf.h
/// \brief Best-First AND/OR Search (AOBF / AO*) with weighted mini-bucket heuristics
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_AOBF_H_
#define IBM_MERLIN_AOBF_H_

#include "aobb.h"
#include "aobf_search_node.h"
#include "aobf_search_space.h"

#include <set>
#include <string>
#include <vector>

namespace merlin {

/**
 * Best-First AND/OR Search (AOBF, a.k.a. AO*) for MAP and Marginal MAP.
 *
 * Tasks supported: MAP, MMAP
 *
 * AOBF explores the context-minimal AND/OR search graph defined by a pseudo tree
 * of the graphical model, guided by the same Weighted Mini-Bucket (WMB) heuristic
 * used by AOBB. Instead of the depth-first branch-and-bound of AOBB, it maintains
 * the current best partial solution tree (following the "best child" markings
 * from the root), repeatedly expands one of its tip nodes, and propagates revised
 * node values bottom-up (Nilsson's algorithm), until the root is solved. Because
 * identical sub-problems are merged in the graph, AOBF never re-expands a node
 * and typically explores far fewer nodes than AOBB, at the cost of holding the
 * explored graph in memory.
 *
 * The search is carried out in COST space (\c -log(probability)): OR nodes take
 * the minimum over children (arc weight + child value), AND nodes take the sum
 * over children, and the algorithm minimizes the root cost -- which maximizes the
 * MAP/MMAP probability. The optimal value is reported in log space, consistent
 * with the other solvers.
 *
 * For MMAP the pseudo tree is constrained so the query (MAX) variables sit at the
 * top; the best-first frontier searches only the MAP variables, and each
 * conditioned SUM sub-problem below a MAP leaf is solved exactly by a nested
 * depth-first AND/OR search.
 *
 * Implemented as a subclass of aobb: it reuses init() (order, pseudo tree, WMB
 * heuristic, cache context) and write_solution(); only the search engine differs.
 */
class aobf: public aobb {
public:

	///
	/// \brief Default constructor.
	///
	aobf() : aobb() {
		m_epsilon = 1.0;
	}

	///
	/// \brief Constructor with a graphical model.
	///
	aobf(const graphical_model& gm) : aobb(gm) {
		m_epsilon = 1.0;
	}

	///
	/// \brief Destructor.
	///
	~aobf() {}

	///
	/// \brief Set the heuristic weight (epsilon). 1.0 => exact AO*; larger values
	///        inflate the heuristic (weighted / greedier search, not exact).
	///
	void set_epsilon(double e) { m_epsilon = e; }
	double get_epsilon() const { return m_epsilon; }

	///
	/// \brief Set the properties of the algorithm.
	///
	/// Adds the Weight property on top of aobb's properties; all other keys are
	/// delegated to aobb::set_properties.
	///
	virtual void set_properties(std::string opt = std::string()) {
		if (opt.length() == 0) {
			set_properties("iBound=10,Order=MinFill,Iter=100,Task=MMAP,"
					"Debug=0,OrderIter=1,Cache=1,TimeLimit=0,Weight=1.0");
			return;
		}
		m_epsilon = 1.0;
		std::vector<std::string> strs = split(opt, ',');
		for (size_t i = 0; i < strs.size(); ++i) {
			std::vector<std::string> asgn = split(strs[i], '=');
			if (asgn.size() == 2 && asgn[0] == "Weight")
				m_epsilon = atof(asgn[1].c_str());
		}
		aobb::set_properties(opt); // parses iBound/Order/Iter/Task/.../TimeLimit
	}

	///
	/// \brief Run the best-first AND/OR search.
	///
	void run();

protected:

	///
	/// \brief Algorithm label used in the solution output.
	///
	virtual const char* algo_name() const { return "aobf"; }

	// --- AO* search engine (mirrors the ground-truth AOBF) ---

	///
	/// \brief The AO* graph search loop: expand best-tip / revise until solved.
	///
	int aostar();

	///
	/// \brief Expand a search node by generating its successors in the graph.
	///        Returns true if the node has no children.
	///
	bool expand(aobf_node* node);

	///
	/// \brief Revise a node's value from its children (OR=min, AND=sum) and mark
	///        it solved when appropriate. Returns true if the value/solved status
	///        changed.
	///
	bool revise(aobf_node* node);

	///
	/// \brief Expand a tip node and propagate revised values up (Nilsson's).
	///
	void expand_and_revise(aobf_node* node);

	///
	/// \brief Rebuild the current best partial solution tree by following the
	///        "best child" markings; collect fringe tip nodes into m_tips and
	///        rebuild the current MAP path assignment. Returns true if unsolved.
	///
	bool find_best_partial_tree();

	///
	/// \brief Heuristic (cost) of an OR node: per value, arc weight and child
	///        cost are cached; OR heur = min over values (MAP) or sum (SUM).
	///
	double heuristic_or_cost(aobf_node* n);

	///
	/// \brief Sort the tip nodes (ascending by heuristic cost).
	///
	void arrange_tip_nodes();

	///
	/// \brief Choose the next tip node to expand (best heuristic).
	///
	aobf_node* choose_tip_node();

	///
	/// \brief Context signature string for a node (var,val) over context vars C.
	///
	std::string context(size_t var, size_t val, const variable_set& C);

	///
	/// \brief Solve the conditioned SUM sub-problem rooted at \c var exactly,
	///        given the current MAP assignment, returning its value (linear).
	///
	double solve_sum(size_t var);

protected:

	// Search state:
	aobf_search_space* m_graph;                 ///< the context-minimal AND/OR graph
	std::vector<aobf_node*> m_tips;             ///< tip nodes of the best partial tree
	std::vector<size_t> m_assignment;           ///< current MAP path assignment (var->val, -1 unset)
	double m_epsilon;                           ///< heuristic weight (1.0 => exact)
	size_t m_global_search_index;               ///< counter for unique root/dummy contexts
	size_t m_num_sum_evals;                     ///< number of SUM sub-problems solved
	std::vector<variable_set> m_and_context;    ///< AND-node context per var (OR ctx + var)
	std::vector<double> m_subtree_norm;         ///< per-var sum of WMB bucket_norm over its
	                                            ///< pseudo-tree subtree (absolute-bound offset)
};

} // namespace merlin

#endif /* IBM_MERLIN_AOBF_H_ */
