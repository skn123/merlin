// Unit tests for the indexing helpers in include/index.h.
// NOTE: index.h uses variable_set but does not include it, so variable_set.h
// must be included first.

#include <gtest/gtest.h>
#include <map>

#include "variable_set.h"
#include "index.h"

using merlin::variable;
using merlin::variable_set;

// index_config and config_index are documented exact inverses.
TEST(Index, ConfigIndexRoundTripBigEndian) {
    variable_set vs(variable(0, 2), variable(1, 3));  // 6 states

    merlin::index_config to_config(vs, /*bigendian=*/true);
    merlin::config_index to_index(vs, /*bigendian=*/true);

    for (size_t i = 0; i < vs.num_states(); ++i) {
        std::map<size_t, size_t> config = to_config.convert(i);
        ASSERT_EQ(config.size(), 2u);
        size_t back = to_index.convert(config);
        EXPECT_EQ(back, i);
    }
}

TEST(Index, ConfigIndexRoundTripLittleEndian) {
    variable_set vs(variable(0, 2), variable(1, 3));

    merlin::index_config to_config(vs, /*bigendian=*/false);
    merlin::config_index to_index(vs, /*bigendian=*/false);

    for (size_t i = 0; i < vs.num_states(); ++i) {
        std::map<size_t, size_t> config = to_config.convert(i);
        size_t back = to_index.convert(config);
        EXPECT_EQ(back, i);
    }
}

TEST(Index, ConfigMapsByVariableLabel) {
    variable_set vs(variable(4, 2), variable(7, 2));  // labels 4 and 7
    merlin::index_config to_config(vs, /*bigendian=*/true);

    std::map<size_t, size_t> config = to_config.convert(0);
    // Config keys must be the actual variable labels.
    EXPECT_NE(config.find(4), config.end());
    EXPECT_NE(config.find(7), config.end());
    EXPECT_EQ(config[4], 0u);
    EXPECT_EQ(config[7], 0u);
}

// subindex is driven by the full set's configuration count; end() equals the
// product of all dimensions in the full set. Note: operator size_t() yields the
// index into the *sub*-scope table (it is not a monotonic 0..end() counter), so
// iteration is driven by end(), exactly as factor::marginal does internally.
TEST(Index, SubindexEndEqualsFullSpace) {
    variable a(0, 2), b(1, 3);
    variable_set full(a, b);           // 6 configs
    variable_set sub(a);               // sub-index over x0
    merlin::subindex s(full, sub);
    EXPECT_EQ(s.end(), full.num_states());
}

TEST(Index, SubindexMapsFullConfigToSubConfig) {
    // full = {x0(2), x1(2)}, sub = {x1}. As we walk the 4 full configs in
    // little-endian order (x0 fastest), the sub-index into the {x1} table must
    // be 0,0,1,1.
    variable x0(0, 2), x1(1, 2);
    variable_set full(x0, x1);
    variable_set sub(x1);

    merlin::subindex s(full, sub);
    size_t expected[4] = {0, 0, 1, 1};
    for (size_t i = 0; i < 4; ++i, ++s)
        EXPECT_EQ((size_t)s, expected[i]) << "at full config " << i;
}
