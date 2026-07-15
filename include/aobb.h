/*
 * aobb.h
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

/// \file aobb.h
/// \brief AND/OR Branch-and-Bound (AOBB) search with weighted mini-bucket heuristics
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_AOBB_H_
#define IBM_MERLIN_AOBB_H_

#include "algorithm.h"
#include "graphical_model.h"
#include "ao_search.h"
#include "wmb.h"

#include <unordered_map>

namespace merlin {

/**
 * AND/OR Branch-and-Bound (AOBB) for MAP and Marginal MAP inference.
 *
 * Tasks supported: MAP, MMAP
 *
 * AOBB is an exact depth-first branch-and-bound search over the AND/OR search
 * tree defined by a pseudo tree of the graphical model. It is guided by a
 * Weighted Mini-Bucket (WMB) heuristic that provides an upper bound on the
 * value of any partial assignment; sub-trees whose bound cannot improve the
 * current best (incumbent) solution are pruned. For MMAP the pseudo tree is
 * constrained so that the query (MAX) variables are eliminated last (i.e., they
 * sit at the top of the tree), which makes the outer maximization valid.
 *
 * The search is carried out in linear (probability) space and the optimal value
 * is reported in log space, consistent with WMB and JGLP.
 */
class aobb: public graphical_model, public algorithm {
public:
	typedef graphical_model::findex findex;        ///< Factor index
	typedef graphical_model::vindex vindex;        ///< Variable index
	typedef graphical_model::flist flist;          ///< Collection of factor indices

public:

	///
	/// \brief Default constructor.
	///
	aobb() : graphical_model() {
		set_properties();
	}

	///
	/// \brief Constructor with a graphical model.
	///
	aobb(const graphical_model& gm) : graphical_model(gm), m_gmo(gm) {
		clear_factors();
		set_properties();
	}

	///
	/// \brief Destructor.
	///
	~aobb() {}

	// Optimization task interface:

	double ub() const { return m_ub; }                     ///< heuristic global upper bound (log)
	double lb() const { return m_logz; }                   ///< value of the best solution found (log)
	std::vector<index> best_config() const { return m_best_config; }

	/// \brief true if the search completed and the reported solution is proven optimal.
	bool is_optimal() const { return m_proved_optimal; }
	/// \brief true if any complete solution was found (false => no solution within the time limit).
	bool found_solution() const { return m_found_solution; }

	// Partition-function interface (not meaningful for search; stubbed):

	double logZ() const { return m_logz; }
	double logZub() const { return m_ub; }
	double logZlb() const { return m_logz; }

	const factor& belief(size_t) const { throw std::runtime_error("Not implemented"); }
	const factor& belief(variable) const { throw std::runtime_error("Not implemented"); }
	const factor& belief(variable_set) const { throw std::runtime_error("Not implemented"); }
	const std::vector<factor>& beliefs() const { throw std::runtime_error("Not implemented"); }

	///
	/// \brief Return the original graphical model.
	///
	const graphical_model& get_gm_orig() const { return m_gmo; }

	///
	/// \brief Inference tasks supported by AOBB.
	///
	MER_ENUM( Task , MAP,MMAP );

	///
	/// \brief Properties of the algorithm.
	///
	MER_ENUM( Property , iBound,Order,Iter,Task,Debug,OrderIter,Cache,TimeLimit,RotateLimit );

public:
	// Setters:

	void set_ibound(size_t i) {
		m_ibound = i ? i : std::numeric_limits<size_t>::max();
	}
	size_t get_ibound() const { return m_ibound; }

	void set_var_types(const std::vector<bool>& vt) { m_var_types = vt; }
	const std::vector<bool>& get_var_types() const { return m_var_types; }

	void set_query(const std::vector<vindex>& q) { m_query = q; }
	const std::vector<vindex>& get_query() { return m_query; }

	void set_order(const variable_order_t& ord) { m_order = ord; }
	const variable_order_t& get_order() { return m_order; }
	void set_order_method(OrderMethod method) {
		m_order.clear();
		m_order_method = method;
	}

	void set_graphical_model(const graphical_model& gm) { m_gmo = gm; }
	void set_graphical_model(const std::vector<factor>& fs) {
		m_gmo = graphical_model(fs);
	}

	///
	/// \brief Set the properties of the algorithm.
	/// \param opt 	The string containing comma separated property value pairs
	///
	virtual void set_properties(std::string opt = std::string()) {
		if (opt.length() == 0) {
			set_properties("iBound=10,Order=MinFill,Iter=100,Task=MMAP,Debug=0,OrderIter=1,Cache=1,TimeLimit=0");
			return;
		}
		m_debug = false;
		m_caching = true;
		m_time_limit = 0.0;
		std::vector<std::string> strs = split(opt, ',');
		for (size_t i = 0; i < strs.size(); ++i) {
			std::vector<std::string> asgn = split(strs[i], '=');
			switch (Property(asgn[0].c_str())) {
			case Property::iBound:
				set_ibound(atol(asgn[1].c_str()));
				break;
			case Property::Order:
				m_order.clear();
				m_parents.clear();
				m_order_method = graphical_model::OrderMethod(asgn[1].c_str());
				break;
			case Property::Iter:
				m_num_iter = atol(asgn[1].c_str());
				break;
			case Property::Task:
				m_task = Task(asgn[1].c_str());
				break;
			case Property::OrderIter:
				m_order_iter = atol(asgn[1].c_str());
				break;
			case Property::Debug:
				m_debug = (atol(asgn[1].c_str()) == 0) ? false : true;
				break;
			case Property::Cache:
				m_caching = (atol(asgn[1].c_str()) == 0) ? false : true;
				break;
			case Property::TimeLimit:
				m_time_limit = atof(asgn[1].c_str());
				break;
			default:
				break;
			}
		}
	}

	///
	/// \brief Initialize the AOBB search (order, pseudo tree, WMB heuristic).
	///
	void init();

	///
	/// \brief Run the AOBB search.
	///
	void run();

	///
	/// \brief Write the solution to the output stream.
	/// \param out				The output stream
	/// \param evidence 		The evidence variable value pairs
	/// \param old2new			The mapping between old and new variable indexing
	/// \param orig 			The graphical model prior to asserting evidence
	/// \param dummies			The dummy (virtual evidence) variables to skip
	/// \param output_format	The output format (json or uai)
	///
	void write_solution(std::ostream& out, const std::map<size_t, size_t>& evidence,
			const std::map<size_t, size_t>& old2new, const graphical_model& orig,
			const std::set<size_t>& dummies, int output_format);

protected:
	// Helpers:

	///
	/// \brief Name of the algorithm, used in the solution output. Overridden by
	///        subclasses (e.g. braobb) so the reused write_solution reports the
	///        right label.
	///
	virtual const char* algo_name() const { return "aobb"; }

	///
	/// \brief Recursively free an AND/OR sub-tree.
	///
	void free_subtree(ao_node* n);

	///
	/// \brief Product of the original factors that become fully instantiated
	///        when \c var is assigned, given the current assignment (linear).
	///
	double label(vindex var, const std::vector<size_t>& asgn) const;

	///
	/// \brief Complete a partial assignment greedily (best label*heuristic per
	///        variable, in pseudo-tree order) and, if the resulting complete
	///        assignment improves the incumbent, record it. Used to keep an
	///        always-available best-so-far solution for the anytime/time-limit
	///        mode. \c partial holds the fixed variables (>=0), others are -1.
	///
	void update_incumbent(const std::vector<size_t>& partial);

	///
	/// \brief Upper bound on the best complete solution consistent with the
	///        partial path from the root down to \c n (linear). Used for pruning.
	///
	double path_bound(ao_node* n) const;

	protected:
	// Members:

	graphical_model m_gmo; 				///< Original graphical model (post-evidence)
	Task m_task;						///< Inference task (MAP or MMAP)
	OrderMethod m_order_method;			///< Variable ordering method
	size_t m_order_iter;				///< Iterations for the ordering heuristic
	size_t m_ibound;					///< Mini-bucket i-bound (heuristic strength)
	size_t m_num_iter;					///< WMB tightening iterations
	double m_logz;						///< log(best value found) => reported value
	double m_ub;						///< Global upper bound from the heuristic (log)
	double m_lb_linear;					///< Best complete-solution value (linear) = incumbent
	variable_order_t m_order;			///< Constrained elimination order
	std::vector<vindex> m_parents;		///< Pseudo tree (parent per variable)
	std::vector<std::vector<vindex> > m_children;	///< Pseudo-tree children
	std::vector<vindex> m_roots;		///< Pseudo-tree roots
	std::vector<bool> m_var_types;		///< true = MAX/query, false = SUM
	std::vector<vindex> m_query;		///< Query (MAX) variables
	std::vector<index> m_best_config;	///< Best assignment (by new/post-evidence index)
	std::vector<flist> m_anchored;		///< Factors anchored at each variable (deepest scope var)
	std::vector<size_t> m_position;		///< m_position[var] = rank in m_order
	wmb m_heuristic;					///< The WMB heuristic engine
	bool m_caching;						///< Enable OR caching (phase 2)
	bool m_debug;						///< Internal debugging flag
	double m_time_limit;				///< Search time limit in seconds (0 = unlimited)
	bool m_proved_optimal;				///< true if the search completed (optimum proved)
	bool m_found_solution;				///< true if any complete solution was found

	// OR caching: per variable, the context (ancestors whose assignment
	// determines the sub-problem below the variable) and a table mapping the
	// context configuration index to the solved sub-tree value.
	std::vector<variable_set> m_cache_context;                   ///< context vars per variable
	/// Per-variable cache: context configuration index -> (solved value, best
	/// sub-assignment achieving it). The sub-assignment is stored so a cache
	/// hit can also restore the MAP/MMAP configuration, not just the value.
	typedef std::pair<double, std::vector<std::pair<vindex, size_t> > > cache_entry;
	std::vector<std::unordered_map<size_t, cache_entry> > m_cache;

	// Search statistics
	size_t m_num_expansions;			///< Number of node expansions
	size_t m_num_cache_hits;			///< Number of OR-cache hits
};

} // namespace merlin

#endif /* IBM_MERLIN_AOBB_H_ */
