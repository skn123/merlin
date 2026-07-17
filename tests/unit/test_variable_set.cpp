// Unit tests for merlin::variable_set and the sub2ind/ind2sub helpers
// (include/variable_set.h).

#include <gtest/gtest.h>
#include <map>
#include <vector>
#include <stdexcept>

#include "variable_set.h"

using merlin::variable;
using merlin::variable_set;

TEST(VariableSet, EmptySetHasOneState) {
    variable_set vs;
    EXPECT_EQ(vs.size(), 0u);
    EXPECT_EQ(vs.nvar(), 0u);
    // num_states() is the empty product = 1.
    EXPECT_EQ(vs.num_states(), 1u);
}

TEST(VariableSet, SingleAndPairConstruction) {
    variable a(0, 2), b(1, 3);
    variable_set single(a);
    EXPECT_EQ(single.size(), 1u);
    EXPECT_EQ(single.num_states(), 2u);

    variable_set pair(a, b);
    EXPECT_EQ(pair.size(), 2u);
    EXPECT_EQ(pair.num_states(), 6u);  // 2 * 3
}

TEST(VariableSet, PairConstructorSortsByLabel) {
    variable a(5, 2), b(1, 4);
    variable_set vs(a, b);  // constructed out of order
    ASSERT_EQ(vs.size(), 2u);
    EXPECT_EQ(vs[0].label(), 1u);  // sorted ascending by label
    EXPECT_EQ(vs[1].label(), 5u);
}

TEST(VariableSet, PairConstructorDeduplicatesEqualLabels) {
    variable a(2, 3), b(2, 3);
    variable_set vs(a, b);
    EXPECT_EQ(vs.size(), 1u);
    EXPECT_EQ(vs.num_states(), 3u);
}

TEST(VariableSet, Union) {
    variable_set x(variable(0, 2));
    variable_set y(variable(1, 3));
    variable_set u = x + y;
    EXPECT_EQ(u.size(), 2u);
    EXPECT_TRUE(u.contains(variable(0, 2)));
    EXPECT_TRUE(u.contains(variable(1, 3)));

    // Union with a single variable.
    variable_set u2 = x | variable(2, 2);
    EXPECT_EQ(u2.size(), 2u);
    EXPECT_TRUE(u2.contains(variable(2, 2)));
}

TEST(VariableSet, Intersection) {
    variable a(0, 2), b(1, 3), c(2, 2);
    variable_set xy(a, b);
    variable_set yc(b, c);
    variable_set inter = xy & yc;
    ASSERT_EQ(inter.size(), 1u);
    EXPECT_EQ(inter[0].label(), 1u);
}

TEST(VariableSet, Difference) {
    variable a(0, 2), b(1, 3), c(2, 2);
    variable_set abc = variable_set(a, b) + variable_set(c);
    variable_set b_only(b);
    variable_set diff = abc - b_only;
    EXPECT_EQ(diff.size(), 2u);
    EXPECT_FALSE(diff.contains(b));
    EXPECT_TRUE(diff.contains(a));
    EXPECT_TRUE(diff.contains(c));
}

TEST(VariableSet, SubsetSupersetAndIntersects) {
    variable a(0, 2), b(1, 3), c(2, 2);
    variable_set ab(a, b);
    variable_set abc = ab + variable_set(c);

    EXPECT_TRUE(ab << abc);   // ab is a subset of abc
    EXPECT_TRUE(abc >> ab);   // abc is a superset of ab
    EXPECT_TRUE(ab.intersects(abc));

    variable_set d(variable(9, 2));
    EXPECT_FALSE(ab.intersects(d));
    EXPECT_FALSE(ab << d);
}

TEST(VariableSet, Contains) {
    variable a(3, 2), b(4, 2);
    variable_set ab(a, b);
    EXPECT_TRUE(ab.contains(a));
    EXPECT_TRUE(ab.contains(b));
    EXPECT_FALSE(ab.contains(variable(5, 2)));
}

// sub2ind/ind2sub use a little-endian layout: the first variable in the set
// is the least significant digit. The map/vector is indexed by variable label.
TEST(VariableSet, Sub2IndVectorRoundTrip) {
    variable a(0, 2), b(1, 3);
    variable_set vs(a, b);  // 6 states

    for (size_t i = 0; i < vs.num_states(); ++i) {
        std::vector<size_t> sub = merlin::ind2sub(vs, i);
        ASSERT_EQ(sub.size(), 2u);
        // Little-endian: i = sub[0] + sub[1] * dims[0].
        EXPECT_EQ(i, sub[0] + sub[1] * 2);
    }
}

TEST(VariableSet, Sub2IndMapRoundTrip) {
    variable a(0, 2), b(1, 3);
    variable_set vs(a, b);

    // NOTE: the MapType& overload of ind2sub in variable_set.h is declared void
    // but returns a value (a latent bug that some compilers reject), so we build
    // the label-keyed config from the vector overload and feed it to sub2ind.
    for (size_t i = 0; i < vs.num_states(); ++i) {
        std::vector<size_t> vsub = merlin::ind2sub(vs, i);  // set-order values
        std::map<size_t, size_t> sub;                       // keyed by label
        for (size_t p = 0; p < vs.size(); ++p)
            sub[vs[p].label()] = vsub[p];
        size_t back = merlin::sub2ind(vs, sub);
        EXPECT_EQ(back, i);
    }
}

TEST(VariableSet, Sub2IndRoundTripThreeVariables) {
    variable a(0, 2), b(1, 3), c(2, 2);
    variable_set vs = variable_set(a, b) + variable_set(c);  // 12 states
    ASSERT_EQ(vs.num_states(), 12u);

    for (size_t i = 0; i < vs.num_states(); ++i) {
        std::vector<size_t> vsub = merlin::ind2sub(vs, i);
        std::map<size_t, size_t> sub;
        for (size_t p = 0; p < vs.size(); ++p)
            sub[vs[p].label()] = vsub[p];
        EXPECT_EQ(merlin::sub2ind(vs, sub), i);
    }
}

TEST(VariableSet, LexicographicComparison) {
    variable_set a(variable(0, 2));
    variable_set b(variable(1, 2));
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a == b);

    variable_set a_copy(variable(0, 2));
    EXPECT_TRUE(a == a_copy);
}
