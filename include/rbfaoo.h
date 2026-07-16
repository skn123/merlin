/*
 * rbfaoo.h
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

/// \file rbfaoo.h
/// \brief Recursive AND/OR Best-First search with Overestimation (RBFAOO)
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_RBFAOO_H_
#define IBM_MERLIN_RBFAOO_H_

#include "aobb.h"
#include "hash_zobrist.h"
#include "rbfaoo_cache_table.h"
#include "rbfaoo_search_node.h"

#include <string>
#include <vector>

namespace merlin {

/**
 * Recursive AND/OR Best-First search with Overestimation (RBFAOO) for MAP/MMAP.
 *
 * Tasks supported: MAP, MMAP
 *
 * RBFAOO [Kishimoto & Marinescu, 2014] is a DFPN-style (depth-first
 * proof-number) recursive best-first search over the context-minimal AND/OR
 * graph, guided by the same Weighted Mini-Bucket (WMB) heuristic as the other
 * AND/OR solvers. Unlike AOBF, which keeps the whole explored graph in memory,
 * RBFAOO stores node values in a fixed-size, Zobrist-hashed transposition table
 * with subtree-size garbage collection, and re-derives values on demand via
 * threshold-controlled iterative deepening -- giving it a small, bounded memory
 * footprint on hard instances. The "overestimation" widens the value threshold
 * around the current best child to reduce thrashing between siblings.
 *
 * The search runs in negative-log COST space (\c -log(probability)): OR nodes
 * over MAX variables take the min, AND nodes sum, OR nodes over SUM variables
 * combine by -logsumexp (marginalization done inline), and the search minimizes
 * the root cost. The WMB heuristic is converted to an absolute per-node cost
 * bound via the subtree-normalization offset (m_subtree_norm), matching AOBB /
 * AOBF. Disproof numbers (dn) track solvedness: dn == DNINF means proven.
 *
 * Implemented as a subclass of aobb: it reuses init() (order, pseudo tree, WMB
 * heuristic, cache context, subtree norm), label() (arc cost), and
 * write_solution(); only the DFPN search engine and its transposition table
 * differ.
 */
class rbfaoo: public aobb {
public:

	rbfaoo() : aobb() { set_defaults(); }
	rbfaoo(const graphical_model& gm) : aobb(gm) { set_defaults(); }
	~rbfaoo() {}

	void set_epsilon(double e) { m_epsilon = e; }
	double get_epsilon() const { return m_epsilon; }
	void set_overestimation(double o) { m_overestimation = o; }
	double get_overestimation() const { return m_overestimation; }
	void set_cache_size(size_t kb) { m_cache_size = kb; }
	size_t get_cache_size() const { return m_cache_size; }

	///
	/// \brief Properties of the algorithm.
	///
	/// Adds Weight / Overestimation / CacheSize on top of aobb's properties; all
	/// other keys are delegated to aobb::set_properties.
	///
	virtual void set_properties(std::string opt = std::string()) {
		if (opt.length() == 0) {
			set_properties("iBound=10,Order=MinFill,Iter=100,Task=MMAP,Debug=0,"
					"OrderIter=1,Cache=1,TimeLimit=0,Weight=1.0,"
					"Overestimation=1.0,CacheSize=1048576");
			return;
		}
		set_defaults();
		std::vector<std::string> strs = split(opt, ',');
		for (size_t i = 0; i < strs.size(); ++i) {
			std::vector<std::string> asgn = split(strs[i], '=');
			if (asgn.size() != 2) continue;
			if (asgn[0] == "Weight")
				m_epsilon = atof(asgn[1].c_str());
			else if (asgn[0] == "Overestimation")
				m_overestimation = atof(asgn[1].c_str());
			else if (asgn[0] == "CacheSize")
				m_cache_size = (size_t) atol(asgn[1].c_str());
		}
		aobb::set_properties(opt); // parses iBound/Order/Iter/Task/.../TimeLimit
	}

	///
	/// \brief Run the recursive best-first AND/OR search.
	///
	void run();

protected:

	virtual const char* algo_name() const { return "rbfaoo"; }

	// --- DFPN recursive best-first engine (mirrors the ground-truth RBFAOO) ---

	/// \brief Top-level recursive search: solve the root, read its cached value.
	int rbfs();

	/// \brief Mutually-recursive iterative deepening on an OR node.
	void mid_or(rbfaoo_node& n, size_t table_index, double th_value,
			int dn_threshold);
	/// \brief Mutually-recursive iterative deepening on an AND node.
	void mid_and(rbfaoo_node& n, size_t table_index, double th_value,
			int dn_threshold);

	/// \brief Generate an AND node's OR children; return the child count and set
	///        best_index (min dn), second_best_dn, solved_flag.
	int generate_children_and(rbfaoo_node& n, int& second_best_dn,
			int& best_index, int& solved_flag);
	/// \brief Recompute an AND node's value/dn from its (cached) children.
	void calculate_and(rbfaoo_node& n, size_t table_index, int num_children,
			int& second_best_dn, int& best_index, int& solved_flag);

	/// \brief Generate an OR node's AND children; return the child count and set
	///        best_index, second_best_value, cutoff_value, solved_flag.
	int generate_children_or(rbfaoo_node& n, double& second_best_value,
			int& best_index, int& solved_flag, double& cutoff_value);
	/// \brief Recompute an OR node's value/dn from its (cached) children.
	void calculate_or(rbfaoo_node& n, size_t table_index, int num_children,
			double& second_best_value, int& best_index, int& solved_flag,
			double& cutoff_value);

	/// \brief Compute the (weighted, absolute) cost heuristic of an OR node and
	///        cache the per-value (total cost, arc cost) pairs for expansion.
	double heuristic_or_cost(rbfaoo_node& n);

	/// \brief Record an AND node's assignment on the current path.
	void do_process(rbfaoo_node* n);

	/// \brief Reconstruct the MAP/MMAP assignment by descending the transposition
	///        table from the root following each MAP OR node's cached best value.
	void reconstruct_assignment();

	/// \brief Sum of disproof numbers, saturating at DNLARGE (DNINF is absorbing).
	int sum_dn(int a, int b) const {
		if (a == DNINF || b == DNINF) return DNINF;
		int d = a + b;
		return (d < (int) DNLARGE) ? d : (int) DNLARGE;
	}

	void set_defaults() {
		m_epsilon = 1.0;
		m_overestimation = 1.0;
		m_cache_size = 1048576; // 1 GB in KB
	}

protected:

	RbfaooCacheTable* m_dfpncache;              ///< the transposition table (TT)
	std::vector<rbfaoo_node*> m_expand;         ///< reusable child-node buffer
	std::vector<size_t> m_assignment;           ///< current path assignment (var->val, -1)
	std::vector<variable_set> m_and_context;    ///< AND-node context (OR ctx + var)
	double m_epsilon;                           ///< heuristic weight (>=1)
	double m_overestimation;                    ///< overestimation for the threshold
	size_t m_cache_size;                        ///< TT budget in kilobytes

	// Statistics:
	size_t m_num_expanded;                      ///< total node expansions
	size_t m_num_expanded_or;
	size_t m_num_expanded_and;
	size_t m_num_sum_evals;                     ///< SUM sub-problems evaluated
	double m_root_result;                       ///< optimal cost read from the TT
};

} // namespace merlin

#endif /* IBM_MERLIN_RBFAOO_H_ */
