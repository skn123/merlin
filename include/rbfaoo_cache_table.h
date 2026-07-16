/*
 * rbfaoo_cache_table.h
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

/// \file rbfaoo_cache_table.h
/// \brief Memory-bounded transposition table for RBFAOO (DFPN cache)
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_RBFAOO_CACHE_TABLE_H_
#define IBM_MERLIN_RBFAOO_CACHE_TABLE_H_

#include "hash_zobrist.h"

#include <cstddef>
#include <vector>

namespace merlin {

/// Context signature (the assigned values of a node's context variables).
typedef std::vector<size_t> rbfaoo_context_t;

///
/// \brief One entry of the RBFAOO transposition table.
///
/// Persists the DFPN state of a solved-or-partially-explored sub-problem so a
/// re-entry can reuse it instead of re-searching. \c subtree_size drives garbage
/// collection: small (cheap-to-recompute) sub-trees are evicted first when the
/// table fills.
///
struct RbfaooCacheElement {
	double result;                  ///< node value (cost, negative-log space)
	RbfaooCacheElement* next;       ///< bucket chain
	rbfaoo_context_t context;       ///< context signature (for exact match)
	Zobrist zobkey;                 ///< Zobrist key (bucket index + fast compare)
	int var;                        ///< node variable
	int best_index;                 ///< best child index (== chosen value for MAP OR)
	int dn;                         ///< disproof number
	unsigned int subtree_size : 30; ///< nodes expanded under this sub-tree (GC key)
	unsigned int or_node_flag : 1;  ///< 1 if OR node, 0 if AND node
	unsigned int solved_flag : 1;   ///< 1 if the sub-tree is solved
};

typedef RbfaooCacheElement* RbfaooCacheElementPtr;

///
/// \brief Fixed-size, Zobrist-hashed transposition table with subtree-size GC.
///
/// The table owns a preallocated pool of elements (bounded by the KB budget) and
/// a bucket array. When the free list is exhausted, small_tree_gc() evicts the
/// entries with the smallest sub-trees (cheapest to recompute) until occupancy
/// drops below a threshold. This bounds RBFAOO's memory to the configured budget.
///
class RbfaooCacheTable {
public:
	RbfaooCacheTable() :
			m_table(NULL), m_entries(NULL), m_table_size(0),
			m_gc_subtree_size(0), m_num_saved(0), m_num_newly_saved(0),
			m_free_list(NULL) {
	}
	~RbfaooCacheTable();

	/// \brief Allocate the table for \c size kilobytes and a given context size.
	void init(size_t size_kb, size_t context_size);

	/// \brief Reset the table to empty (keeps the allocation).
	void clear();

	/// \brief Look up an entry by (Zobrist, var, or_flag, context). NULL if absent.
	RbfaooCacheElementPtr read(const Zobrist& zob, int var, int or_flag,
			const rbfaoo_context_t& ctxt);

	/// \brief Insert or update an entry. \c ctxt is swapped in on insertion.
	RbfaooCacheElementPtr write(const Zobrist& zob, int var,
			rbfaoo_context_t& ctxt, int or_flag, unsigned int best_index,
			double result, int dn, int solved_flag,
			unsigned int increased_subtree_size);

	size_t num_entries() const { return m_table_size; }

private:
	void small_tree_gc();
	size_t table_index(const Zobrist& zob) const {
		return zob.getC0() % m_table_size;
	}

	RbfaooCacheElement** m_table;      ///< bucket heads
	RbfaooCacheElement* m_entries;     ///< preallocated element pool
	size_t m_table_size;               ///< number of buckets / pool size
	size_t m_gc_subtree_size;          ///< current GC eviction threshold
	size_t m_num_saved;                ///< total writes (stats)
	size_t m_num_newly_saved;          ///< new inserts (stats)
	RbfaooCacheElement* m_free_list;   ///< free element list
};

} // namespace merlin

#endif /* IBM_MERLIN_RBFAOO_CACHE_TABLE_H_ */
