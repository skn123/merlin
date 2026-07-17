/*
 * gls.h
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

/// \file gls.h
/// \brief Guided Local Search (GLS+) for MAP inference
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_GLS_H_
#define IBM_MERLIN_GLS_H_

#include "sls.h"

#include <string>
#include <vector>

namespace merlin {

/**
 * Guided Local Search (GLS+) for MAP inference.
 *
 * Tasks supported: MAP
 *
 * GLS+ [Hutter, Hoos & Stützle, IJCAI-05] augments greedy local search with
 * factor-entry penalties to escape local minima without random restarts. At each
 * step it takes the greedy best flip by the augmented score
 * \c logProbScore + penaltyScore. When it reaches a local minimum (no improving
 * flip), it increases the penalty of the currently-instantiated factor entries
 * with maximal "utility" (\c -logCost / (1 + penalty), preferring high-cost,
 * lightly-penalized entries), which reshapes the search surface. Penalties are
 * periodically scaled down (smoothed) so old penalties fade.
 *
 * Implemented as a subclass of \c sls: it reuses the incremental-scoring engine
 * (log-factor tables, occurrence lists, Markov blankets, flip_to, best_new_inst,
 * good-variable maintenance) and only adds the penalty tables, the augmented
 * score, and the guided run loop. All values are in negative-log (log-prob)
 * space, consistent with SLS and the AND/OR solvers.
 */
class gls: public sls {
public:

	gls() : sls() { set_gls_defaults(); }
	gls(const graphical_model& gm) : sls(gm) { set_gls_defaults(); }
	virtual ~gls() {}

	///
	/// \brief Properties of the algorithm (SLS keys plus GLS+ keys).
	///
	MER_ENUM( Property , Task,TimeLimit,Iter,Seed,Debug,
			PenaltyIncrement,PenaltyMultFactor,Smooth,Interval );

	///
	/// \brief Set the properties of the algorithm.
	///
	/// Parses the GLS+ keys, then delegates the shared keys to sls::set_properties.
	///
	virtual void set_properties(std::string opt = std::string()) {
		if (opt.length() == 0) {
			set_properties("Task=MAP,TimeLimit=10,Iter=100000000,Seed=12345678,"
					"Debug=0,PenaltyIncrement=1.0,PenaltyMultFactor=10000,"
					"Smooth=0.999,Interval=200");
			return;
		}
		set_gls_defaults();
		std::vector<std::string> strs = split(opt, ',');
		for (size_t i = 0; i < strs.size(); ++i) {
			std::vector<std::string> asgn = split(strs[i], '=');
			if (asgn.size() != 2) continue;
			if (asgn[0] == "PenaltyIncrement")
				m_penalty_increment = atof(asgn[1].c_str());
			else if (asgn[0] == "PenaltyMultFactor")
				m_penalty_mult = atof(asgn[1].c_str());
			else if (asgn[0] == "Smooth")
				m_smooth = atof(asgn[1].c_str());
			else if (asgn[0] == "Interval")
				m_interval = atol(asgn[1].c_str());
		}
		sls::set_properties(opt); // parses Task/TimeLimit/Iter/Seed/Debug
	}

	void init();
	void run();

protected:

	virtual const char* algo_name() const { return "gls+"; }

	///
	/// \brief Augmented score: log-prob delta plus the penalty delta (weighted).
	///
	/// m_pen_score[var][val] holds the raw penalty delta (penalty[new] -
	/// penalty[old]); flipping toward a lower-penalty entry (negative delta)
	/// should raise the score, hence the -m_penalty_mult weight.
	///
	virtual double score(vindex var, size_t val) const {
		return m_log_score[var][val] - m_penalty_mult * m_pen_score[var][val];
	}

	///
	/// \brief MMAP score of flipping \c var to \c val: the MMAP objective delta
	///        (from sls::score_mmap) minus the guided-search penalty delta over
	///        the MAP factors the flipped variable touches. Penalties live on the
	///        factor entries (m_penalty) and are read on demand.
	///
	virtual double score_mmap(vindex var, size_t val);

	/// \brief Penalty delta (weighted) of flipping \c var to \c val, from the
	///        factors it occurs in: -m_penalty_mult * Σ (penalty[new] - penalty[old]).
	double penalty_delta(vindex var, size_t val) const;

	/// \brief At a local minimum, increase the penalty of the max-utility entries.
	void increase_penalties();
	/// \brief Scale all penalties toward zero (smoothing).
	void scale_penalties(double factor);
	/// \brief Utility of factor f's current entry: -logCost / (1 + penalty).
	double utility(findex f) const;

	void set_gls_defaults() {
		m_penalty_increment = 1.0;
		m_penalty_mult = 10000.0;
		m_smooth = 0.999;
		m_interval = 200;
	}

protected:

	double m_penalty_increment;   ///< penalty added per local-minimum update
	double m_penalty_mult;        ///< weight making penalties commensurate with log-prob
	double m_smooth;              ///< periodic penalty-scaling factor (<1)
	long m_interval;              ///< scale penalties every this many local minima
};

} // namespace merlin

#endif /* IBM_MERLIN_GLS_H_ */
