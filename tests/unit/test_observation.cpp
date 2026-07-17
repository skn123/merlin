// Unit tests for merlin::observation (include/observation.h).

#include <gtest/gtest.h>
#include <vector>
#include "observation.h"

using merlin::observation;

TEST(Observation, DefaultIsNeitherObservedNorVirtual) {
    observation o;
    EXPECT_FALSE(o.is_observed());
    EXPECT_FALSE(o.is_virtual());
}

TEST(Observation, MissingValueConstructor) {
    observation o(3);  // variable 3, value missing
    EXPECT_EQ(o.var(), 3);
    EXPECT_FALSE(o.is_observed());
    EXPECT_FALSE(o.is_virtual());
}

TEST(Observation, PlainEvidenceConstructor) {
    observation o(2, 1);  // variable 2 observed at value 1
    EXPECT_EQ(o.var(), 2);
    EXPECT_EQ(o.val(), 1);
    EXPECT_TRUE(o.is_observed());
    EXPECT_FALSE(o.is_virtual());
}

TEST(Observation, VirtualEvidenceConstructor) {
    std::vector<double> lik;
    lik.push_back(0.6);
    lik.push_back(0.4);
    observation o(5, lik);
    EXPECT_EQ(o.var(), 5);
    EXPECT_FALSE(o.is_observed());
    EXPECT_TRUE(o.is_virtual());

    std::vector<double> got = o.likelihood();
    ASSERT_EQ(got.size(), 2u);
    EXPECT_DOUBLE_EQ(got[0], 0.6);
    EXPECT_DOUBLE_EQ(got[1], 0.4);
}
