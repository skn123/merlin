/*
 * rbfaoo_cache_table.cpp
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

/// \file rbfaoo_cache_table.cpp
/// \brief Memory-bounded transposition table for RBFAOO (DFPN cache)
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#include "rbfaoo_cache_table.h"

#include <algorithm>

namespace merlin {

RbfaooCacheTable::~RbfaooCacheTable() {
	if (m_table) {
		delete[] m_table;
		delete[] m_entries;
		m_table = NULL;
		m_entries = NULL;
	}
}

void RbfaooCacheTable::init(size_t size_kb, size_t context_size) {
	if (m_table) {
		delete[] m_table;
		delete[] m_entries;
		m_free_list = NULL;
	}
	// Number of entries that fit the KB budget, accounting for the element, its
	// bucket pointer, and the per-entry context storage.
	size_t per_entry = sizeof(RbfaooCacheElement) + sizeof(RbfaooCacheElementPtr)
			+ context_size * sizeof(size_t);
	m_table_size = (size_kb * 1024) / per_entry;
	if (m_table_size < 1024) m_table_size = 1024; // sane minimum

	m_entries = new RbfaooCacheElement[m_table_size];
	m_table = new RbfaooCacheElementPtr[m_table_size];
	std::fill(&m_table[0], &m_table[m_table_size], (RbfaooCacheElementPtr) NULL);

	m_free_list = m_entries;
	for (size_t i = 0; i < m_table_size - 1; ++i)
		m_entries[i].next = &m_entries[i + 1];
	m_entries[m_table_size - 1].next = NULL;
	m_gc_subtree_size = 0;
}

void RbfaooCacheTable::clear() {
	if (m_table && m_entries) {
		std::fill(&m_table[0], &m_table[m_table_size], (RbfaooCacheElementPtr) NULL);
		m_free_list = m_entries;
		for (size_t i = 0; i < m_table_size - 1; ++i)
			m_entries[i].next = &m_entries[i + 1];
		m_entries[m_table_size - 1].next = NULL;
		m_gc_subtree_size = 0;
	}
}

// Evict the entries with the smallest sub-trees (cheapest to recompute), raising
// the eviction threshold until occupancy drops below TABLE_OCCUPANCY. This bounds
// the table to its allocated size; evicted sub-problems are recomputed on demand.
void RbfaooCacheTable::small_tree_gc() {
	const double TABLE_OCCUPANCY = 0.70;
	size_t num_stored = m_table_size; // table is full when GC triggers
	double occupancy;
	rbfaoo_context_t empty_context;

	do {
		unsigned int smallest_subtree = 0x7fffffff;
		for (size_t i = 0; i < m_table_size; ++i) {
			RbfaooCacheElementPtr prev = NULL, element = m_table[i];
			while (element != NULL) {
				if (element->subtree_size <= m_gc_subtree_size) {
					// Evict: release its context storage and return it to the pool.
					element->context.swap(empty_context);
					empty_context.clear();
					if (prev == NULL)
						m_table[i] = element->next;
					else
						prev->next = element->next;
					--num_stored;
					RbfaooCacheElementPtr tmp = element;
					element = element->next;
					tmp->next = m_free_list;
					m_free_list = tmp;
				} else {
					smallest_subtree = std::min(smallest_subtree,
							(unsigned int) element->subtree_size);
					prev = element;
					element = element->next;
				}
			}
		}
		occupancy = (double) num_stored / (double) m_table_size;
		m_gc_subtree_size = smallest_subtree;
	} while (occupancy >= TABLE_OCCUPANCY);
}

RbfaooCacheElementPtr RbfaooCacheTable::read(const Zobrist& zob, int var,
		int or_flag, const rbfaoo_context_t& ctxt) {
	size_t index = table_index(zob);
	for (RbfaooCacheElement* e = m_table[index]; e != NULL; e = e->next) {
		if (zob == e->zobkey && var == e->var
				&& (int) e->or_node_flag == or_flag && e->context == ctxt)
			return e;
	}
	return NULL;
}

RbfaooCacheElementPtr RbfaooCacheTable::write(const Zobrist& zob, int var,
		rbfaoo_context_t& ctxt, int or_flag, unsigned int best_index,
		double result, int dn, int solved_flag,
		unsigned int increased_subtree_size) {
	size_t index = table_index(zob);
	++m_num_saved;

	// Update an existing entry if present.
	for (RbfaooCacheElement* e = m_table[index]; e != NULL; e = e->next) {
		if (zob == e->zobkey && var == e->var
				&& (int) e->or_node_flag == or_flag && e->context == ctxt) {
			e->best_index = best_index;
			e->result = result;
			e->dn = dn;
			e->solved_flag = solved_flag;
			e->subtree_size += increased_subtree_size;
			return e;
		}
	}

	// Insert a new entry (GC first if the pool is exhausted).
	if (m_free_list == NULL)
		small_tree_gc();

	RbfaooCacheElementPtr e = m_free_list;
	++m_num_newly_saved;
	m_free_list = m_free_list->next;
	e->zobkey = zob;
	e->subtree_size = increased_subtree_size;
	e->next = m_table[index];
	m_table[index] = e;
	e->var = var;
	e->result = result;
	e->dn = dn;
	e->solved_flag = solved_flag;
	e->context.swap(ctxt);
	e->best_index = best_index;
	e->or_node_flag = or_flag;
	return e;
}

} // namespace merlin
