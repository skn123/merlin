/*
 * rbfaoo_search_node.h
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

/// \file rbfaoo_search_node.h
/// \brief Transient search node for RBFAOO (recursive best-first AND/OR search)
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_RBFAOO_SEARCH_NODE_H_
#define IBM_MERLIN_RBFAOO_SEARCH_NODE_H_

#include "ao_search.h"          // AO_OR / AO_AND
#include "hash_zobrist.h"
#include "rbfaoo_cache_table.h" // rbfaoo_context_t

#include <cstddef>
#include <limits>
#include <vector>

namespace merlin {

/// Disproof-number constants (proof-number search bookkeeping).
enum {
	DNINF = 1 << 30,    ///< "infinite" disproof number => solved / proven
	DNLARGE = 1 << 28,  ///< cap for summed disproof numbers
	UNDETERMINED = -1   ///< no best child chosen yet
};

///
/// \brief A transient node used while expanding/evaluating during RBFAOO.
///
/// RBFAOO keeps only a small window of nodes alive at a time (the current
/// expansion frontier in \c m_expand); the persistent value/dn/solved state
/// lives in the transposition table. Each node carries its Zobrist key and
/// context so it can be looked up / written. Values are COSTS (negative-log
/// space): OR/MAX = min child cost, AND = sum, OR/SUM = -logsumexp.
///
class rbfaoo_node {
public:
	rbfaoo_node(int node_type, size_t var, size_t val, int depth,
			double label = 0.0) :
			m_node_value(std::numeric_limits<double>::quiet_NaN()),
			m_heur(std::numeric_limits<double>::quiet_NaN()),
			m_label(label), m_var(var), m_val(val), m_depth(depth),
			m_type(node_type), m_dn(0), m_solved(0), m_heur_cache(NULL) {
	}
	// OR-node constructor (no value/label).
	rbfaoo_node(int node_type, size_t var, int depth) :
			m_node_value(std::numeric_limits<double>::quiet_NaN()),
			m_heur(std::numeric_limits<double>::quiet_NaN()),
			m_label(0.0), m_var(var), m_val(0), m_depth(depth),
			m_type(node_type), m_dn(0), m_solved(0), m_heur_cache(NULL) {
	}
	~rbfaoo_node() {
		if (m_type == AO_OR && m_heur_cache)
			delete[] m_heur_cache;
	}

	int type() const { return m_type; }
	size_t var() const { return m_var; }
	size_t val() const { return m_val; }
	int depth() const { return m_depth; }

	void set_node_value(double d) { m_node_value = d; }
	double node_value() const { return m_node_value; }
	void set_heur(double d) { m_heur = d; }
	double heur() const { return m_heur; }
	double label() const { return m_label; }
	void set_label(double d) { m_label = d; }

	int dn() const { return m_dn; }
	void set_dn(int d) { m_dn = d; }
	int solved() const { return m_solved; }
	void set_solved(int s) { m_solved = s; }

	// OR-node per-value cache: [2*v] = child total cost (incl. arc), [2*v+1] = arc.
	void set_heur_cache(double* d) { m_heur_cache = d; }
	double* heur_cache() const { return m_heur_cache; }

	Zobrist& zobrist() { return m_zobrist; }
	const Zobrist& zobrist() const { return m_zobrist; }
	rbfaoo_context_t& context() { return m_context; }
	const rbfaoo_context_t& context() const { return m_context; }
	void set_context(const rbfaoo_context_t& c) { m_context = c; }
	void add_context(size_t v) { m_context.push_back(v); }

private:
	double m_node_value;   ///< node value (cost)
	double m_heur;         ///< heuristic estimate (cost)
	double m_label;        ///< AND: arc label cost; OR: unused
	size_t m_var;
	size_t m_val;          ///< AND: assigned value
	int m_depth;
	int m_type;            ///< AO_OR / AO_AND
	int m_dn;              ///< disproof number
	int m_solved;          ///< 0 = unsolved, 1 = solved
	double* m_heur_cache;  ///< OR only: per-value (total, arc) costs
	Zobrist m_zobrist;
	rbfaoo_context_t m_context;
};

} // namespace merlin

#endif /* IBM_MERLIN_RBFAOO_SEARCH_NODE_H_ */
