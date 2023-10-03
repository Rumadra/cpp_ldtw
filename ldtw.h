#include <Eigen/Dense>
#include "unsupported/Eigen/CXX11/Tensor"
#include <vector>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

// Œ^’è‹`
using Pair = std::pair<int, int>;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using IntVector = Eigen::VectorXi;
using Path = std::vector<std::pair<int, int>>;

// Function prototypes
double local_squared_dist(const std::vector<double>& x, const std::vector<double>& y); Matrix to_time_series(const Eigen::VectorXd& ts, bool remove_nans = false);
int max_steps(int i, int j, int max_length, int length_1, int length_2);
std::map<Pair, std::map<int, double>> limited_warping_length_cost(const std::vector<double>& s1, const std::vector<double>& s2, int max_length);
std::vector<Pair> return_path_limited_warping_length(
    const std::map<Pair, std::map<int, double>>& accum_costs,
    const Pair& target_indices,
    int optimal_length
);

// pybind API
std::pair<std::vector<Pair>, double> dtw_path_limited_warping_length(
    const std::vector<double>& s1,
    const std::vector<double>& s2,
    int max_length
);