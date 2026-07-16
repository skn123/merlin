/*
 * hash_zobrist.h
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

/// \file hash_zobrist.h
/// \brief Zobrist hash keys for the RBFAOO transposition table
/// \author Radu Marinescu radu.marinescu@ie.ibm.com

#ifndef IBM_MERLIN_HASH_ZOBRIST_H_
#define IBM_MERLIN_HASH_ZOBRIST_H_

#include "graphical_model.h"
#include "variable_set.h"

#include <cstdint>
#include <vector>

namespace merlin {

///
/// \brief A Zobrist hash key over an AND/OR search state.
///
/// The key is a pair (c0: 64-bit, c1: 32-bit) formed by XOR-ing per-(variable,
/// value) random constants over the variables in an OR node's context, then
/// perturbed by the assigned value for the corresponding AND node. Two states
/// hashing to the same key are (with overwhelming probability) identical; the
/// transposition table still compares the full context to guarantee correctness.
///
/// The random tables are shared across all Zobrist instances and are seeded once
/// per solve via init_once() with a fixed seed (deterministic runs).
///
class Zobrist {
public:
	typedef graphical_model::vindex vindex;

	Zobrist() : m_c0(0), m_c1(0) {}

	uint64_t getC0() const { return m_c0; }
	uint32_t getC1() const { return m_c1; }

	bool operator==(const Zobrist& other) const {
		return m_c0 == other.m_c0 && m_c1 == other.m_c1;
	}

	///
	/// \brief Encode an OR node: XOR the per-(var,val) randoms over the context
	///        variables at their currently-assigned values.
	/// \param vars   the OR node's context variables (their values must be set)
	/// \param asgn   the current path assignment (var -> value)
	///
	void encodeOR(const variable_set& vars, const std::vector<size_t>& asgn);

	///
	/// \brief Encode an AND node from its parent OR node's key and the value.
	///
	void encodeAND(const Zobrist& z, size_t value) {
		*this = z;
		m_c0 += (uint64_t) value + 1;
	}

	/// \brief Seed the shared random tables from the model (fixed seed). Call once.
	static void init_once(const graphical_model& gm);
	/// \brief Release the shared random tables.
	static void finish_once();

private:
	uint64_t m_c0;
	uint32_t m_c1;

	static std::vector<std::vector<uint64_t> > c0_table; // [var][val]
	static std::vector<std::vector<uint32_t> > c1_table; // [var][val]
	static bool initialized;
};

} // namespace merlin

#endif /* IBM_MERLIN_HASH_ZOBRIST_H_ */
