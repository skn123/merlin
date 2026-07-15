// End-to-end integration tests: drive the Merlin facade on the committed
// cancer.uai fixtures and compare against the golden outputs committed at the
// repository root (cancer.uai.MAR, cancer.uai.EM).
//
// Only the exact CTE algorithm is used, so results are deterministic.

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include "merlin.h"
#include "base.h"
#include "graphical_model.h"

namespace {

std::string data_path(const char* name) {
    return std::string(MERLIN_DATA_DIR) + "/" + name;
}
std::string fixtures_path(const char* name) {
    return std::string(MERLIN_FIXTURES_DIR) + "/" + name;
}
std::string tmp_path(const char* name) {
    return std::string(MERLIN_TEST_TMP_DIR) + "/" + name;
}

// Read the whole file as a string.
std::string slurp(const std::string& path) {
    std::ifstream is(path.c_str());
    std::ostringstream ss;
    ss << is.rdbuf();
    return ss.str();
}

}  // namespace

// ---- MAR / PR against cancer.uai.MAR ----------------------------------------

TEST(Integration, CancerMARMatchesGolden) {
    const std::string out_base = tmp_path("cancer_mar_out");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_evidence_file(data_path("cancer.evid"));
    eng.set_task(MERLIN_TASK_MAR);
    eng.set_algorithm(MERLIN_ALGO_CTE);  // exact inference
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    // Merlin appends ".MAR" to the output file for the MAR task.
    std::string produced = slurp(out_base + ".MAR");
    ASSERT_FALSE(produced.empty()) << "no MAR output produced";

    // Parse: token "PR" then the log-value, ... token "MAR" then
    // <nvar> then per-variable <states> <p0..p_{states-1}>.
    std::istringstream is(produced);
    std::string tok;
    double logZ = 0.0;
    bool got_pr = false;
    std::vector<double> marginals;
    size_t nvar = 0;

    while (is >> tok) {
        if (tok == "PR") {
            is >> logZ;
            got_pr = true;
        } else if (tok == "MAR") {
            is >> nvar;
            for (size_t v = 0; v < nvar; ++v) {
                size_t states = 0;
                is >> states;
                for (size_t k = 0; k < states; ++k) {
                    double p = 0.0;
                    is >> p;
                    marginals.push_back(p);
                }
            }
        }
    }

    ASSERT_TRUE(got_pr);
    ASSERT_EQ(nvar, 5u);
    ASSERT_EQ(marginals.size(), 10u);  // 5 binary variables

    // Golden values from cancer.uai.MAR.
    EXPECT_NEAR(logZ, -1.139434, 1e-4);
    const double expected[10] = {
        0.5,   0.5,     // var 0
        1.0,   0.0,     // var 1 (evidence)
        0.125, 0.875,   // var 2
        0.8,   0.2,     // var 3
        0.625, 0.375    // var 4
    };
    for (size_t i = 0; i < 10; ++i)
        EXPECT_NEAR(marginals[i], expected[i], 1e-4) << "marginal index " << i;
}

// ---- EM regression against a pinned snapshot -------------------------------
//
// NOTE: the golden committed at the repo root, cancer.uai.EM, has every CPT
// entry at 0.5. The current EM implementation does NOT reproduce it -- it
// learns real parameters from cancer.dat. That root file is a stale artifact
// predating the EM fixes in the git history ("Fixed a bug in EM virtual
// evidence"), so it is not a valid regression target.
//
// Instead we pin against tests/fixtures/cancer.expected.EM, a snapshot of the
// current (correct) EM output. This locks in behavior: if EM changes, this test
// flags it and the snapshot can be re-reviewed and updated deliberately.
TEST(Integration, CancerEMMatchesPinnedSnapshot) {
    const std::string out_base = tmp_path("cancer_em_out");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_dataset_file(data_path("cancer.dat"));
    eng.set_task(MERLIN_TASK_EM);
    eng.set_algorithm(MERLIN_ALGO_CTE);
    eng.set_iterations(10);
    eng.set_threshold(1e-6);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced_path = out_base + ".EM";
    std::ifstream produced_is(produced_path.c_str());
    ASSERT_TRUE(produced_is.good()) << "no EM output at " << produced_path;
    merlin::graphical_model learned;
    learned.read(produced_is);

    std::ifstream expected_is(fixtures_path("cancer.expected.EM").c_str());
    ASSERT_TRUE(expected_is.good());
    merlin::graphical_model expected;
    expected.read(expected_is);

    EXPECT_EQ(learned.nvar(), expected.nvar());
    ASSERT_EQ(learned.num_factors(), expected.num_factors());

    // Same model structure, so factors line up by index. Compare cell-for-cell.
    for (size_t f = 0; f < learned.num_factors(); ++f) {
        const merlin::factor& lf = learned.get_factor(f);
        const merlin::factor& ef = expected.get_factor(f);
        ASSERT_EQ(lf.num_states(), ef.num_states());
        // Tolerance is loose enough (5e-3) to absorb floating-point drift from
        // different optimization levels across the 10 EM iterations, while
        // still catching any behavioral change in the learning.
        for (size_t k = 0; k < lf.num_states(); ++k)
            EXPECT_NEAR(lf[k], ef[k], 5e-3) << "factor " << f << " cell " << k;
    }

    // Sanity: EM actually learned (did not leave everything at the trivial 0.5).
    bool learned_nontrivial = false;
    for (size_t f = 0; f < learned.num_factors() && !learned_nontrivial; ++f) {
        const merlin::factor& lf = learned.get_factor(f);
        for (size_t k = 0; k < lf.num_states(); ++k)
            if (std::fabs(lf[k] - 0.5) > 1e-3) { learned_nontrivial = true; break; }
    }
    EXPECT_TRUE(learned_nontrivial);
}

// ---- Regression: CTE MAR on a model with more variables than factors --------
//
// paskin.uai has 6 variables but only 5 factors. CTE previously sized its
// elimination graph by num_nodes() (= factor count = 5), so the variable with
// label 5 indexed one past the end of the adjacency table -- a heap overflow
// that crashed non-deterministically (SIGSEGV / std::length_error). This guards
// against a regression: the run must complete cleanly and yield 6 marginals.
TEST(Integration, PaskinCTEMARDoesNotCrash) {
    const std::string out_base = tmp_path("paskin_mar_out");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("paskin.uai"));
    eng.set_task(MERLIN_TASK_MAR);
    eng.set_algorithm(MERLIN_ALGO_CTE);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAR");
    ASSERT_FALSE(produced.empty()) << "no MAR output produced";

    std::istringstream is(produced);
    std::string tok;
    size_t nvar = 0;
    std::vector<double> marginals;
    while (is >> tok) {
        if (tok == "MAR") {
            is >> nvar;
            for (size_t v = 0; v < nvar; ++v) {
                size_t states = 0;
                is >> states;
                for (size_t k = 0; k < states; ++k) {
                    double p = 0.0;
                    is >> p;
                    marginals.push_back(p);
                }
            }
        }
    }

    EXPECT_EQ(nvar, 6u);
    ASSERT_EQ(marginals.size(), 12u);  // 6 binary variables
    // Every marginal is a valid probability, and each pair sums to 1.
    for (size_t v = 0; v < 6; ++v) {
        double p0 = marginals[2 * v], p1 = marginals[2 * v + 1];
        EXPECT_GE(p0, 0.0); EXPECT_LE(p0, 1.0);
        EXPECT_GE(p1, 0.0); EXPECT_LE(p1, 1.0);
        EXPECT_NEAR(p0 + p1, 1.0, 1e-6) << "variable " << v;
    }
}

// ---- AOBB (exact AND/OR branch-and-bound) MAP / MMAP ------------------------
//
// AOBB is an exact optimizer, so its reported value must equal the true optimum
// (verified here against values computed by brute force over the joint):
//   cancer.uai MAP  (evidence var1=0): log value -2.617844, config 1 0 1 0 0
//   simple5.uai MMAP (query 0 1 2)   : log value 11.285626, config 1 1 0

TEST(Integration, CancerMAPAobbIsExact) {
    const std::string out_base = tmp_path("cancer_map_aobb");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_evidence_file(data_path("cancer.evid"));
    eng.set_task(MERLIN_TASK_MAP);
    eng.set_algorithm(MERLIN_ALGO_AOBB);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAP");
    ASSERT_FALSE(produced.empty()) << "no MAP output produced";

    // Parse: token "MAP", then <count> followed by one value per variable.
    std::istringstream is(produced);
    std::string tok;
    size_t count = 0;
    std::vector<size_t> cfg;
    while (is >> tok) {
        if (tok == "MAP") {
            is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); }
        }
    }
    ASSERT_EQ(count, 5u);
    ASSERT_EQ(cfg.size(), 5u);
    const size_t expected[5] = {1, 0, 1, 0, 0};  // brute-force MAP config
    for (size_t i = 0; i < 5; ++i)
        EXPECT_EQ(cfg[i], expected[i]) << "variable " << i;
}

TEST(Integration, Simple5MMAPAobbIsExact) {
    const std::string out_base = tmp_path("simple5_mmap_aobb");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("simple5.uai"));
    eng.set_query_file(data_path("simple5.map"));
    eng.set_task(MERLIN_TASK_MMAP);
    eng.set_algorithm(MERLIN_ALGO_AOBB);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MMAP");
    ASSERT_FALSE(produced.empty()) << "no MMAP output produced";

    std::istringstream is(produced);
    std::string tok;
    size_t count = 0;
    std::vector<size_t> cfg;
    while (is >> tok) {
        if (tok == "MMAP") {
            is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); }
        }
    }
    ASSERT_EQ(count, 3u);
    ASSERT_EQ(cfg.size(), 3u);
    const size_t expected[3] = {1, 1, 0};  // brute-force MMAP query config
    for (size_t i = 0; i < 3; ++i)
        EXPECT_EQ(cfg[i], expected[i]) << "query variable " << i;
}

// A generous time limit does not change the result: the search still completes
// and reports the proven optimum with an "optimal" status in the JSON output.
TEST(Integration, CancerMAPAobbTimeLimitStillOptimal) {
    const std::string out_base = tmp_path("cancer_map_aobb_tl");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_evidence_file(data_path("cancer.evid"));
    eng.set_task(MERLIN_TASK_MAP);
    eng.set_algorithm(MERLIN_ALGO_AOBB);
    eng.set_time_limit(60.0);                 // ample: search completes well within
    eng.set_output_format(MERLIN_OUTPUT_JSON);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAP.json");
    ASSERT_FALSE(produced.empty()) << "no MAP JSON output produced";
    // Completed search => proven optimal.
    EXPECT_NE(produced.find("\"optimal\" : true"), std::string::npos)
        << "expected optimal:true in: " << produced;
    EXPECT_NE(produced.find("\"status\" : \"true\""), std::string::npos);
    // The optimum value is unchanged by the (unreached) time limit.
    EXPECT_NE(produced.find("\"value\" : -2.617844"), std::string::npos)
        << produced;
}
