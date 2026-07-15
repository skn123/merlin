// Unit tests for merlin::variable (include/variable.h).

#include <gtest/gtest.h>
#include "variable.h"

using merlin::variable;

TEST(Variable, DefaultConstructor) {
    variable v;
    EXPECT_EQ(v.label(), 0u);
    EXPECT_EQ(v.states(), 0u);
}

TEST(Variable, LabelAndStates) {
    variable v(3, 4);
    EXPECT_EQ(v.label(), 3u);
    EXPECT_EQ(v.states(), 4u);
}

TEST(Variable, ImplicitSizeTConversionReturnsLabel) {
    variable v(7, 2);
    size_t as_index = v;  // operator size_t()
    EXPECT_EQ(as_index, 7u);
}

TEST(Variable, ComparisonsOrderByLabelNotStates) {
    variable a(1, 5);
    variable b(2, 2);
    variable a_same_label(1, 99);  // different states, same label

    EXPECT_LT(a, b);
    EXPECT_LE(a, b);
    EXPECT_GT(b, a);
    EXPECT_GE(b, a);
    EXPECT_NE(a, b);

    // Equality is by label only.
    EXPECT_EQ(a, a_same_label);
    EXPECT_TRUE(a <= a_same_label);
    EXPECT_TRUE(a >= a_same_label);
    EXPECT_FALSE(a < a_same_label);
}
