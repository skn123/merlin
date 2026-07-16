/*
 * aobf_search_node.h
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

/// \file aobf_search_node.h
/// \brief Nodes of the context-minimal AND/OR graph used by best-first search
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_AOBF_SEARCH_NODE_H_
#define IBM_MERLIN_AOBF_SEARCH_NODE_H_

#include "ao_search.h"      // ao_node_type: AO_OR / AO_AND

#include <cassert>
#include <cstddef>
#include <limits>
#include <list>

namespace merlin {

///
/// \brief A node in the context-minimal AND/OR search graph (AOBF).
///
/// Unlike the depth-first ao_node (a tree node with a single parent), an AOBF
/// node lives in a context-minimal *graph*: identical sub-problems are merged, so
/// an AND node may have several parents. Values are kept in cost space
/// (\c -log(probability)): OR nodes take the MIN over children (best value +
/// arc weight), AND nodes take the SUM over children. The search minimizes the
/// root cost, which maximizes the MAP/MMAP probability.
///
/// This is the base class; concrete nodes are aobf_and_node / aobf_or_node.
///
class aobf_node {
protected:
	int m_type;                          ///< AO_OR or AO_AND
	int m_depth;                         ///< depth in the pseudo tree
	size_t m_var;                        ///< node variable (X_i)
	double m_value;                      ///< node value == cost (a lower bound)
	double m_heur;                       ///< heuristic estimate of the value (cost)
	double m_upper_bound;                ///< best solution cost found below (unused for exact)
	int m_index;                         ///< revision counter (Nilsson's expandAndRevise)

	std::list<aobf_node*> m_parents;     ///< parents (AND may have several; OR one)
	std::list<aobf_node*> m_children;    ///< children in the AND/OR graph
	aobf_node* m_current_parent;         ///< parent in the current best partial tree

	// Boolean flags (kept as plain bools for readability).
	bool m_fringe;                       ///< on the fringe (not yet expanded)
	bool m_solved;                       ///< sub-graph rooted here is solved
	bool m_terminal;                     ///< terminal node (no children)
	bool m_expanded;                     ///< children already generated
	bool m_visited;                      ///< transient mark used during revision
	bool m_deadend;                      ///< dead end (infinite cost)

public:
	aobf_node() :
			m_type(AO_OR), m_depth(0), m_var(0),
			m_value(std::numeric_limits<double>::quiet_NaN()),
			m_heur(std::numeric_limits<double>::quiet_NaN()),
			m_upper_bound(std::numeric_limits<double>::infinity()),
			m_index(0), m_current_parent(NULL), m_fringe(false),
			m_solved(false), m_terminal(false), m_expanded(false),
			m_visited(false), m_deadend(false) {
	}
	virtual ~aobf_node() {
		m_parents.clear();
		m_children.clear();
	}

	// Type / identity:
	virtual int type() const = 0;                 ///< AO_OR or AO_AND
	virtual size_t val() const = 0;               ///< AND: assigned value (OR: undefined)
	size_t var() const { return m_var; }
	int depth() const { return m_depth; }

	// Value / heuristic (cost space):
	void set_value(double v) { m_value = v; }
	double value() const { return m_value; }
	void set_heur(double h) { m_heur = h; }
	double heur() const { return m_heur; }
	void set_upper_bound(double u) { m_upper_bound = u; }
	double upper_bound() const { return m_upper_bound; }

	// Revision index (Nilsson's algorithm):
	void set_index(int i) { m_index = i; }
	int index() const { return m_index; }
	void inc_index() { ++m_index; }
	void dec_index() { --m_index; }

	// Graph structure:
	void add_child(aobf_node* n) { m_children.push_back(n); }
	void add_parent(aobf_node* n) { m_parents.push_back(n); }
	const std::list<aobf_node*>& children() const { return m_children; }
	std::list<aobf_node*>& children() { return m_children; }
	const std::list<aobf_node*>& parents() const { return m_parents; }
	std::list<aobf_node*>& parents() { return m_parents; }

	void set_current_parent(aobf_node* n) { m_current_parent = n; }
	aobf_node* current_parent() const { return m_current_parent; }

	// Best child (OR nodes only):
	virtual void set_best_child(aobf_node* n) = 0;
	virtual aobf_node* best_child() const = 0;

	// Per-value heuristic cache (OR nodes only). Layout: for domain value v,
	// cache[2*v]   = cost of the AND child (includes the arc weight),
	// cache[2*v+1] = arc weight (== -log(label)).
	virtual void set_heur_cache(double* d) = 0;
	virtual double* heur_cache() const = 0;
	virtual void clear_heur_cache() = 0;
	virtual double weight(size_t v) const = 0;

	// Flags:
	void set_fringe(bool f) { m_fringe = f; }
	bool is_fringe() const { return m_fringe; }
	void set_solved(bool f) { m_solved = f; }
	bool is_solved() const { return m_solved; }
	void set_terminal(bool f) { m_terminal = f; }
	bool is_terminal() const { return m_terminal; }
	void set_expanded(bool f) { m_expanded = f; }
	bool is_expanded() const { return m_expanded; }
	void set_visited(bool f) { m_visited = f; }
	bool is_visited() const { return m_visited; }
	void set_deadend(bool f) { m_deadend = f; }
	bool is_deadend() const { return m_deadend; }
};

///
/// \brief AND node: a value assignment <X_i, a> in the AND/OR graph.
///
class aobf_and_node : public aobf_node {
protected:
	size_t m_val;   ///< the value assigned to the OR-parent variable

public:
	aobf_and_node(size_t var, size_t val, int depth) {
		m_type = AO_AND;
		m_var = var;
		m_val = val;
		m_depth = depth;
	}
	~aobf_and_node() {}

	int type() const { return AO_AND; }
	size_t val() const { return m_val; }

	// AND nodes carry no best child / heuristic cache / weights.
	void set_best_child(aobf_node*) { assert(false); }
	aobf_node* best_child() const { assert(false); return NULL; }
	void set_heur_cache(double*) { assert(false); }
	double* heur_cache() const { assert(false); return NULL; }
	void clear_heur_cache() {}
	double weight(size_t) const { assert(false); return 0.0; }
};

///
/// \brief OR node: a variable choice for X_i in the AND/OR graph.
///
class aobf_or_node : public aobf_node {
protected:
	aobf_node* m_best_child;   ///< best AND child (marks the best partial tree)
	double* m_heur_cache;      ///< per-value (cost, weight) pairs; size 2*domain

public:
	aobf_or_node(size_t var, int depth) : m_best_child(NULL), m_heur_cache(NULL) {
		m_type = AO_OR;
		m_var = var;
		m_depth = depth;
	}
	~aobf_or_node() { clear_heur_cache(); }

	int type() const { return AO_OR; }
	size_t val() const { assert(false); return (size_t) -1; } // no value for OR

	void set_best_child(aobf_node* n) { m_best_child = n; }
	aobf_node* best_child() const { return m_best_child; }

	void set_heur_cache(double* d) { m_heur_cache = d; }
	double* heur_cache() const { return m_heur_cache; }
	void clear_heur_cache() {
		if (m_heur_cache) { delete[] m_heur_cache; m_heur_cache = NULL; }
	}
	void set_weight(size_t v, double w) {
		assert(m_heur_cache != NULL);
		m_heur_cache[2 * v + 1] = w;
	}
	double weight(size_t v) const {
		assert(m_heur_cache != NULL);
		return m_heur_cache[2 * v + 1];
	}
};

} // namespace merlin

#endif /* IBM_MERLIN_AOBF_SEARCH_NODE_H_ */
