/*
 * aobf_search_space.h
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

/// \file aobf_search_space.h
/// \brief The context-minimal AND/OR search graph used by best-first search
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_AOBF_SEARCH_SPACE_H_
#define IBM_MERLIN_AOBF_SEARCH_SPACE_H_

#include "aobf_search_node.h"

#include <cassert>
#include <string>
#include <unordered_map>

namespace merlin {

///
/// \brief A search state = the identity of a node in the context-minimal graph.
///
/// Two nodes are the same iff they have the same type (AND/OR) and the same
/// context string (the instantiation of the node's context variables). Merging
/// identical states is what turns the AND/OR tree into a (smaller) graph.
///
struct aobf_state {
	int type;             ///< AO_OR or AO_AND
	std::string context;  ///< context signature string

	aobf_state(int t, const std::string& s) : type(t), context(s) {}
	bool operator==(const aobf_state& a) const {
		return type == a.type && context == a.context;
	}
};

/// \brief Hasher for aobf_state (default string hash on the context signature).
struct aobf_state_hash {
	size_t operator()(const aobf_state& a) const {
		return std::hash<std::string>()(a.context) ^ (size_t) a.type;
	}
};

/// The context-minimal AND/OR graph: state -> node.
typedef std::unordered_map<aobf_state, aobf_node*, aobf_state_hash> aobf_graph;

///
/// \brief The context-minimal AND/OR search graph.
///
/// Owns every node; deleting the space frees all nodes. Nodes are indexed by
/// their aobf_state so that identical sub-problems generated on different paths
/// map to the same node (graph merging).
///
class aobf_search_space {
protected:
	aobf_graph m_nodes;     ///< the graph (state -> node)
	aobf_node* m_root;      ///< root node (an OR node)
	size_t m_and_nodes;     ///< number of AND nodes expanded
	size_t m_or_nodes;      ///< number of OR nodes expanded

public:
	aobf_search_space() : m_root(NULL), m_and_nodes(0), m_or_nodes(0) {}
	~aobf_search_space() { clear(); }

	// Root:
	aobf_node* root() const { return m_root; }
	void set_root(aobf_node* n) { m_root = n; }

	// Graph operations:
	void add(const aobf_state& state, aobf_node* node) {
		assert(node != NULL);
		m_nodes.insert(std::make_pair(state, node));
	}
	bool find(const aobf_state& state) const {
		return m_nodes.find(state) != m_nodes.end();
	}
	aobf_node* get(const aobf_state& state) {
		aobf_graph::iterator it = m_nodes.find(state);
		return (it != m_nodes.end()) ? it->second : NULL;
	}
	void erase(const aobf_state& state) {
		aobf_graph::iterator it = m_nodes.find(state);
		if (it != m_nodes.end()) {
			aobf_node* n = it->second;
			m_nodes.erase(it);
			delete n;
		}
	}
	void clear() {
		for (aobf_graph::iterator it = m_nodes.begin(); it != m_nodes.end(); ++it)
			delete it->second;
		m_nodes.clear();
		m_root = NULL;
	}

	// Statistics:
	size_t and_nodes() const { return m_and_nodes; }
	size_t or_nodes() const { return m_or_nodes; }
	void inc_expanded(int node_type) {
		if (node_type == AO_AND) ++m_and_nodes; else ++m_or_nodes;
	}

private:
	aobf_search_space(const aobf_search_space&);            // no copy
	aobf_search_space& operator=(const aobf_search_space&); // no assign
};

} // namespace merlin

#endif /* IBM_MERLIN_AOBF_SEARCH_SPACE_H_ */
