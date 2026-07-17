/*
 * hash_zobrist.cpp
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

/// \file hash_zobrist.cpp
/// \brief Zobrist hash keys for the RBFAOO transposition table
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#include "hash_zobrist.h"

#include <cstdlib>

namespace merlin {

std::vector<std::vector<uint64_t> > Zobrist::c0_table;
std::vector<std::vector<uint32_t> > Zobrist::c1_table;
bool Zobrist::initialized = false;

void Zobrist::init_once(const graphical_model& gm) {
	size_t n = gm.nvar();
	c0_table.assign(n, std::vector<uint64_t>());
	c1_table.assign(n, std::vector<uint32_t>());
	// Fixed seed => deterministic runs (mirrors the ground truth's srandom(1)).
	std::srand(1);
	for (size_t v = 0; v < n; ++v) {
		size_t d = gm.var(v).states();
		c0_table[v].resize(d);
		c1_table[v].resize(d);
		for (size_t a = 0; a < d; ++a) {
			// Combine two rand() draws into a 64-bit constant.
			uint64_t hi = (uint64_t) std::rand();
			uint64_t lo = (uint64_t) std::rand();
			c0_table[v][a] = (hi << 31) | lo;
			c1_table[v][a] = (uint32_t) std::rand();
		}
	}
	initialized = true;
}

void Zobrist::finish_once() {
	c0_table.clear();
	c1_table.clear();
	initialized = false;
}

void Zobrist::encodeOR(const variable_set& vars, const std::vector<size_t>& asgn) {
	m_c0 = 0;
	m_c1 = 0;
	for (variable_set::const_iterator it = vars.begin(); it != vars.end(); ++it) {
		vindex var = it->label();
		size_t val = asgn[var];
		// Context variables must be assigned on the current path.
		if (val == (size_t) -1) val = 0; // defensive (cardinality-1 / unset)
		if (var < c0_table.size() && val < c0_table[var].size()) {
			m_c0 ^= c0_table[var][val];
			m_c1 ^= c1_table[var][val];
		}
	}
}

} // namespace merlin
