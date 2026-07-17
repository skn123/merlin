/*
 * sls.h
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

/// \file sls.h
/// \brief Stochastic Local Search (SLS / G+StS) for MAP inference
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_SLS_H_
#define IBM_MERLIN_SLS_H_

#include "algorithm.h"
#include "graphical_model.h"
#include "wmb.h"

#include <vector>

namespace merlin {

/**
 * Stochastic Local Search (SLS) for MAP inference.
 *
 * Tasks supported: MAP
 *
 * This is the "G+StS" (Greedy + Stochastic Simulation) algorithm from Hutter,
 * Hoos & Stützle (IJCAI-05), adapted to Merlin. It explores complete variable
 * assignments by single-variable "flips": with probability \c noise a random
 * variable is flipped to a value sampled proportionally to its score, otherwise
 * the greedy best-improving flip is taken; the search restarts on stagnation.
 *
 * The objective is the log-probability of a complete assignment (sum of the log
 * of the instantiated factor values, natural log), which the search MAXIMIZES --
 * the same quantity the AND/OR solvers report as \c m_logz. This is negative-log
 * space in the sense that factor values are combined additively as logs. Zero-
 * probability factor entries are clamped to a large finite negative value
 * (LOG_ZERO) to avoid -inf arithmetic.
 *
 * Efficiency comes from incremental scoring over the Markov blanket: each
 * variable caches, per value, the change in log-probability from flipping to
 * that value; a flip updates only the flipped variable and its neighbors
 * (variables sharing a factor with it), and the set of "good" variables (those
 * with an improving flip) is maintained incrementally.
 *
 * GLS+ (class \c gls) subclasses this and adds guided-local-search penalties;
 * the shared incremental-scoring engine lives here.
 */
class sls: public graphical_model, public algorithm {
public:
	typedef graphical_model::findex findex;
	typedef graphical_model::vindex vindex;
	typedef graphical_model::flist flist;

public:

	sls() : graphical_model() { set_properties(); }
	sls(const graphical_model& gm) : graphical_model(gm), m_gmo(gm) {
		clear_factors();
		set_properties();
	}
	virtual ~sls() {}

	// Optimization-task interface:
	double ub() const { return std::numeric_limits<double>::infinity(); } // no bound
	double lb() const { return m_logz; }
	std::vector<index> best_config() const { return m_best_config; }
	bool is_optimal() const { return false; }         // SLS never proves optimality
	bool found_solution() const { return m_found_solution; }

	// Partition-function interface (not meaningful for local search; stubbed):
	double logZ() const { return m_logz; }
	double logZub() const { return std::numeric_limits<double>::infinity(); }
	double logZlb() const { return m_logz; }
	const factor& belief(size_t) const { throw std::runtime_error("Not implemented"); }
	const factor& belief(variable) const { throw std::runtime_error("Not implemented"); }
	const factor& belief(variable_set) const { throw std::runtime_error("Not implemented"); }
	const std::vector<factor>& beliefs() const { throw std::runtime_error("Not implemented"); }

	const graphical_model& get_gm_orig() const { return m_gmo; }

	///
	/// \brief Inference tasks supported by SLS.
	///
	MER_ENUM( Task , MAP,MMAP );

	///
	/// \brief Properties of the algorithm.
	///
	MER_ENUM( Property , Task,TimeLimit,Iter,Seed,Noise,Cutoff,iBound,Debug );

public:
	// Setters:
	void set_query(const std::vector<vindex>& q) { m_query = q; }
	const std::vector<vindex>& get_query() { return m_query; }
	void set_graphical_model(const graphical_model& gm) { m_gmo = gm; }

	///
	/// \brief Set the properties of the algorithm.
	///
	virtual void set_properties(std::string opt = std::string()) {
		if (opt.length() == 0) {
			set_properties("Task=MAP,TimeLimit=10,Iter=100000000,Seed=12345678,"
					"Noise=40,Cutoff=5.0,iBound=10,Debug=0");
			return;
		}
		set_defaults();
		std::vector<std::string> strs = split(opt, ',');
		for (size_t i = 0; i < strs.size(); ++i) {
			std::vector<std::string> asgn = split(strs[i], '=');
			if (asgn.size() != 2) continue;
			switch (Property(asgn[0].c_str())) {
			case Property::Task:      m_task = Task(asgn[1].c_str()); break;
			case Property::TimeLimit: m_time_limit = atof(asgn[1].c_str()); break;
			case Property::Iter:      m_max_flips = atol(asgn[1].c_str()); break;
			case Property::Seed:      m_seed = (size_t) atol(asgn[1].c_str()); break;
			case Property::Noise:     m_noise = atol(asgn[1].c_str()); break;
			case Property::Cutoff:    m_cutoff = atof(asgn[1].c_str()); break;
			case Property::iBound:    m_ibound = (size_t) atol(asgn[1].c_str()); break;
			case Property::Debug:     m_debug = (atol(asgn[1].c_str()) != 0); break;
			default: break;
			}
		}
	}

	void init();
	void run();

	///
	/// \brief Write the MAP solution to the output stream (JSON or UAI).
	///
	void write_solution(std::ostream& out, const std::map<size_t, size_t>& evidence,
			const std::map<size_t, size_t>& old2new, const graphical_model& orig,
			const std::set<size_t>& dummies, int output_format);

protected:

	virtual const char* algo_name() const { return "sls"; }

	// --- Shared incremental-scoring engine ---

	/// \brief Build the per-factor log tables, strides, occurrence lists, and MB.
	void build_structures();
	/// \brief Full (from-scratch) recompute of scores/good-vars/log_prob (cancels
	///        accumulated floating-point drift; also the initial build after init).
	void recompute_scores();
	/// \brief Incrementally flip \c var to \c val, updating log_prob, factor
	///        indices, and the scores of \c var and its Markov-blanket neighbors.
	void flip_to(vindex var, size_t val);
	/// \brief Best improving (var,val) over the good variables; returns var==-1
	///        if none (a local minimum). Random tie-break.
	bool best_new_inst(vindex& out_var, size_t& out_val);
	/// \brief Sample a new value for \c var proportionally to exp(score) (skips
	///        the current value).
	size_t sampled_new_val(vindex var);
	/// \brief A uniformly random value for \c var different from its current one.
	size_t random_new_val(vindex var);
	/// \brief A uniformly random variable that the local search may flip (all
	///        variables for MAP; only the MAP/query variables for MMAP).
	vindex random_search_var();
	/// \brief Update the incumbent (best-so-far) assignment / log-prob.
	void update_best();
	/// \brief Recompute whether \c var is a "good" variable and update the set.
	void update_good_var(vindex var);

	///
	/// \brief Score of flipping \c var to \c val. SLS uses the log-prob delta;
	///        GLS+ overrides to add the penalty delta. (MAP objective; the MMAP
	///        path scores via score_mmap instead.)
	///
	virtual double score(vindex var, size_t val) const {
		return m_log_score[var][val];
	}

	// --- Marginal MAP (MMAP) support ---

	/// \brief Build the WMB heuristic (constrained order, var types, pseudo tree,
	///        subtree-norm) and the SUM-root list used to estimate the conditioned
	///        partition of the SUM variables. Called from build_structures() when
	///        the task is MMAP.
	void build_mmap_structures();
	/// \brief Estimate Q(x_M) = log P(x_M, evidence) for the current MAP
	///        assignment in \c asgn (MAP vars set, SUM vars == -1): the exact
	///        MAP-factor log-contributions plus the WMB conditioned-SUM estimate
	///        over the SUM-root subtrees. Returned in log-probability space.
	double eval_mmap(std::vector<size_t>& asgn);
	/// \brief WMB conditioned-SUM cost estimate for the subtree rooted at SUM
	///        variable \c x, given the MAP ancestors in \c asgn (cost space).
	///        Mirrors aobb::heuristic_or for a SUM node.
	double sum_estimate(vindex x, std::vector<size_t>& asgn);
	/// \brief Score of flipping MAP variable \c var to \c val under the MMAP
	///        objective: Q(with var=val) - Q(current). Used when m_task == MMAP.
	///        GLS+ overrides this to subtract the guided-search penalty delta.
	virtual double score_mmap(vindex var, size_t val);

	/// \brief Random reinitialization of the assignment + score recompute.
	void random_restart();

	void set_defaults() {
		m_task = Task::MAP;
		m_time_limit = 10.0;
		m_max_flips = 100000000L;
		m_seed = 12345678;
		m_noise = 40;
		m_cutoff = 5.0;
		m_ibound = 10;
		m_debug = false;
	}

	/// \brief Whether the local search should continue (time/flip budget).
	bool ls_continue() const {
		if (m_max_flips > 0 && m_num_flips >= m_max_flips) return false;
		if (m_time_limit > 0.0 &&
				(timeSystem() - m_start_time) >= m_time_limit) return false;
		return true;
	}

protected:
	// Members:

	graphical_model m_gmo;              ///< Original model (post-evidence)
	Task m_task;                        ///< Inference task (MAP)
	double m_logz;                      ///< log of best value found (reported)
	std::vector<index> m_best_config;   ///< best assignment (post-evidence index)
	std::vector<vindex> m_query;        ///< query (MAP) variables (all non-evidence)
	bool m_found_solution;              ///< a (finite) solution was found
	bool m_debug;

	// Parameters:
	double m_time_limit;                ///< wall-clock budget (seconds; 0 = none)
	long m_max_flips;                   ///< flip budget (0 = none)
	size_t m_seed;                      ///< RNG seed
	int m_noise;                        ///< noise probability (percent) for G+StS
	double m_cutoff;                    ///< stagnation-restart cutoff multiplier
	size_t m_ibound;                    ///< i-bound for the MMAP WMB heuristic

	double LOG_ZERO;                    ///< clamp for log(0) (large finite negative)

	// Current search state:
	std::vector<size_t> m_value;        ///< current assignment (by variable id)
	double m_log_prob;                  ///< current assignment log-probability
	long m_num_flips;                   ///< flips performed this run

	// Per-factor structures (indexed by factor id):
	std::vector<std::vector<double> > m_log_factor; ///< log of each factor entry
	std::vector<std::vector<double> > m_penalty;    ///< GLS+ penalty per entry (else empty)
	std::vector<std::vector<size_t> > m_scope;      ///< factor scope var labels (vs order)
	std::vector<std::vector<size_t> > m_stride;     ///< index stride per scope position
	std::vector<size_t> m_index;                    ///< current linear index of each factor

	// Per-variable structures (indexed by variable id):
	std::vector<size_t> m_dom;                      ///< domain size
	std::vector<std::vector<double> > m_log_score;  ///< Δlog-prob if flipped to val
	std::vector<std::vector<double> > m_pen_score;  ///< Δpenalty if flipped to val (GLS+)
	std::vector<std::vector<std::pair<findex, size_t> > > m_occ; ///< (factor, local pos) list
	std::vector<std::vector<vindex> > m_mb;         ///< Markov blanket (incl. self)

	// Good-variable set (variables with a strictly improving flip):
	std::vector<vindex> m_good_vars;
	std::vector<char> m_is_good;

	// --- MMAP-only structures ---
	std::vector<char> m_is_map;         ///< per variable: true if a MAP/query var
	std::vector<vindex> m_map_vars;     ///< the MAP variables (local search flips these)
	wmb m_heuristic;                    ///< WMB heuristic for the conditioned-SUM estimate
	std::vector<bool> m_var_types;      ///< per variable: true = MAP/query, false = SUM
	variable_order_t m_order;           ///< constrained elimination order (SUM first)
	std::vector<vindex> m_parents;      ///< pseudo-tree parent per variable
	std::vector<std::vector<vindex> > m_children; ///< pseudo-tree children
	std::vector<double> m_subtree_norm; ///< per-var WMB bucket_norm summed over subtree
	std::vector<vindex> m_sum_roots;    ///< SUM vars whose pseudo-tree parent is MAP/root
	std::vector<flist> m_anchored;      ///< factors anchored at each variable (deepest scope var)
	std::vector<size_t> m_position;     ///< rank of each variable in m_order
	std::vector<char> m_is_map_factor;  ///< per factor: true if its scope is all MAP vars

	// Statistics:
	long m_num_flips_total;             ///< flips across all restarts
	long m_num_restarts;
	long m_num_evals;                   ///< MMAP objective evaluations (WMB estimator calls)
};

} // namespace merlin

#endif /* IBM_MERLIN_SLS_H_ */
