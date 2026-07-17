// Unit tests for the MER_ENUM type-safe enum macro (include/enum.h).

#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>

#include "enum.h"

namespace {
// Instantiate an enum inside the test translation unit.
MER_ENUM(Color, Red, Green, Blue);
}  // namespace

TEST(MerEnum, ValueConstructionAndImplicitType) {
    Color c(Color::Green);
    EXPECT_EQ(c, Color::Green);
    Color::Type t = c;
    EXPECT_EQ(t, Color::Green);
}

TEST(MerEnum, DefaultsToFirstValue) {
    Color c;
    EXPECT_EQ(c, Color::Red);
}

TEST(MerEnum, StringConstruction) {
    // The name buffer is built by stringizing the macro arguments, so the
    // first value ("Red") and the value immediately after a comma ("Green")
    // match by prefix.
    Color red("Red");
    EXPECT_EQ(red, Color::Red);
    Color green("Green");
    EXPECT_EQ(green, Color::Green);
}

// KNOWN QUIRK: because __VA_ARGS__ stringizes with a space after each comma,
// the internal names buffer is "Red,Green, Blue,". The const char* constructor
// prefix-matches from a comma boundary, so the third and later values are
// stored with a leading space and only match when queried with that space.
// This documents current behavior; it is a latent bug in MER_ENUM, not a
// property tests should depend on for round-tripping 3rd+ values.
TEST(MerEnum, ThirdValueRequiresLeadingSpace_KnownQuirk) {
    EXPECT_THROW(Color("Blue"), std::runtime_error);
    Color blue(" Blue");
    EXPECT_EQ(blue, Color::Blue);
}

TEST(MerEnum, ToCStringRoundTrip) {
    Color c(Color::Green);
    const char* s = c;  // operator const char*
    Color back(s);
    EXPECT_EQ(back, Color::Green);
}

TEST(MerEnum, UnknownStringThrows) {
    EXPECT_THROW(Color("Purple"), std::runtime_error);
}

TEST(MerEnum, StreamExtractionFirstValues) {
    std::istringstream is("Green");
    Color c;
    is >> c;
    EXPECT_EQ(c, Color::Green);
}
