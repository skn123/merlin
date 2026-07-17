/*
 * braobb.h
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

/// \file braobb.h
/// \brief Breadth-Rotating AND/OR Branch-and-Bound (BRAOBB) search
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_BRAOBB_H_
#define IBM_MERLIN_BRAOBB_H_

#include "aobb.h"

namespace merlin {

/**
 * Breadth-Rotating AND/OR Branch-and-Bound (BRAOBB) for MAP and MMAP.
 *
 * Tasks supported: MAP, MMAP
 *
 * BRAOBB [Otten & Dechter, 2012] explores the same AND/OR search space as AOBB
 * with the same weighted mini-bucket heuristic, OR caching and branch-and-bound
 * pruning, but instead of a single depth-first stack it maintains a FIFO queue
 * of independent subproblem stacks and rotates between them: after a fixed
 * number of node expansions ("rotate limit") on one subproblem it moves to the
 * next. This reaches leaves across many subproblems early and therefore
 * produces much better anytime (best-so-far) solutions on hard instances. The
 * proven optimum is identical to AOBB; only the exploration order and the
 * quality of intermediate solutions differ.
 *
 * Implemented as a subclass of aobb: it reuses init(), the WMB heuristic,
 * label()/path_bound()/update_incumbent(), the OR cache, and the anytime /
 * time-limit machinery unchanged; only run() (the search frontier) differs.
 */
class braobb: public aobb {
public:

	///
	/// \brief Default constructor.
	///
	braobb() : aobb() {
		m_rotate_limit = 1000;
	}

	///
	/// \brief Constructor with a graphical model.
	///
	braobb(const graphical_model& gm) : aobb(gm) {
		m_rotate_limit = 1000;
	}

	///
	/// \brief Destructor.
	///
	~braobb() {}

	///
	/// \brief Set the rotation limit (node expansions per subproblem before
	///        rotating to the next; 0 disables rotation).
	///
	void set_rotate_limit(size_t r) { m_rotate_limit = r; }
	size_t get_rotate_limit() const { return m_rotate_limit; }

	///
	/// \brief Set the properties of the algorithm.
	///
	/// Adds the RotateLimit property on top of aobb's properties; all other
	/// keys are delegated to aobb::set_properties.
	///
	virtual void set_properties(std::string opt = std::string()) {
		if (opt.length() == 0) {
			set_properties("iBound=10,Order=MinFill,Iter=100,Task=MMAP,"
					"Debug=0,OrderIter=1,Cache=1,TimeLimit=0,RotateLimit=1000");
			return;
		}
		// Extract our RotateLimit key, then delegate everything else to aobb.
		m_rotate_limit = 1000;
		std::vector<std::string> strs = split(opt, ',');
		for (size_t i = 0; i < strs.size(); ++i) {
			std::vector<std::string> asgn = split(strs[i], '=');
			if (asgn.size() == 2 && asgn[0] == "RotateLimit")
				m_rotate_limit = (size_t) atol(asgn[1].c_str());
		}
		aobb::set_properties(opt); // parses iBound/Order/Iter/Task/.../TimeLimit
	}

	///
	/// \brief Run the breadth-rotating AND/OR branch-and-bound search.
	///
	void run();

protected:

	///
	/// \brief Algorithm label used in the solution output.
	///
	virtual const char* algo_name() const { return "braobb"; }

protected:

	size_t m_rotate_limit;   ///< Node expansions per subproblem before rotating (0 = no rotation)
};

} // namespace merlin

#endif /* IBM_MERLIN_BRAOBB_H_ */
