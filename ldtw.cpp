#include "ldtw.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

// Compute the local squared distance between two vectors
double local_squared_dist(const std::vector<double>& x, const std::vector<double>& y) {
    double dist = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double diff = x[i] - y[i];
        dist += diff * diff;
    }
    return dist;
}

// TODO •s—v
// Transform a time series to the required format
Matrix to_time_series(const Eigen::VectorXd& ts, bool remove_nans) {
    // For this implementation, we assume that NaN removal is not required
    // as Eigen does not have native NaN support like cupy.
    Matrix ts_out = ts;
    if (ts_out.rows() == 1) {
        ts_out.transposeInPlace();
    }
    return ts_out;
}

// Compute the maximum number of steps required to reach a given cell in L-DTW
int max_steps(int i, int j, int max_length, int length_1, int length_2) {
    int candidate_1 = i + j;
    int candidate_2 = max_length - std::max(length_1 - i - 1, length_2 - j - 1);
    return std::min(candidate_1, candidate_2);
}

// Return the DTW path given the cost matrix and optimal length
std::vector<Pair> return_path_limited_warping_length(
    const std::map<Pair, std::map<int, double>>& accum_costs,
    const Pair& target_indices,
    int optimal_length
) {
    std::vector<Pair> path = { target_indices };
    int cur_length = optimal_length;

    while (path.back() != Pair(0, 0)) {
        int i = path.back().first;
        int j = path.back().second;

        if (i == 0) {
            path.push_back({ 0, j - 1 });
        }
        else if (j == 0) {
            path.push_back({ i - 1, 0 });
        }
        else {
            double val1 = accum_costs.at({ i - 1, j - 1 }).count(cur_length - 1) ? accum_costs.at({ i - 1, j - 1 }).at(cur_length - 1) : std::numeric_limits<double>::infinity();
            double val2 = accum_costs.at({ i - 1, j }).count(cur_length - 1) ? accum_costs.at({ i - 1, j }).at(cur_length - 1) : std::numeric_limits<double>::infinity();
            double val3 = accum_costs.at({ i, j - 1 }).count(cur_length - 1) ? accum_costs.at({ i, j - 1 }).at(cur_length - 1) : std::numeric_limits<double>::infinity();

            if (val1 <= val2 && val1 <= val3) {
                path.push_back({ i - 1, j - 1 });
            }
            else if (val2 <= val1 && val2 <= val3) {
                path.push_back({ i - 1, j });
            }
            else {
                path.push_back({ i, j - 1 });
            }

            cur_length--;
        }
    }

    std::reverse(path.begin(), path.end());
    return path;
}

// Compute the cost matrix for limited warping length DTW
std::map<Pair, std::map<int, double>> limited_warping_length_cost(const std::vector<double>& s1, const std::vector<double>& s2, int max_length) {
    std::map<Pair, std::map<int, double>> dict_costs;

    for (int i = 0; i < s1.size(); ++i) {
        for (int j = 0; j < s2.size(); ++j) {
            dict_costs[{i, j}] = {};
        }
    }

    // Init
    dict_costs[{0, 0}][0] = local_squared_dist({ s1[0] }, { s2[0] });
    for (int i = 1; i < s1.size(); ++i) {
        double pred = dict_costs[{i - 1, 0}][i - 1];
        dict_costs[{i, 0}][i] = pred + local_squared_dist({ s1[i] }, { s2[0] });
    }
    for (int j = 1; j < s2.size(); ++j) {
        double pred = dict_costs[{0, j - 1}][j - 1];
        dict_costs[{0, j}][j] = pred + local_squared_dist({ s1[0] }, { s2[j] });
    }

    // Main loop
    for (int i = 1; i < s1.size(); ++i) {
        for (int j = 1; j < s2.size(); ++j) {
            int min_s = std::max(i, j);
            int max_s = max_steps(i, j, max_length - 1, s1.size(), s2.size());
            for (int s = min_s; s <= max_s; ++s) {
                dict_costs[{i, j}][s] = local_squared_dist({ s1[i] }, { s2[j] });
                dict_costs[{i, j}][s] += std::min(
                    dict_costs[{i, j - 1}][s - 1],
                    std::min(
                        dict_costs[{i - 1, j}][s - 1],
                        dict_costs[{i - 1, j - 1}][s - 1]
                    )
                );
            }
        }
    }

    return dict_costs;
}

// Compute the DTW path with limited warping length
std::pair<std::vector<Pair>, double> dtw_path_limited_warping_length(
    const std::vector<double>& s1,
    const std::vector<double>& s2,
    int max_length
) {
    if (max_length < std::max(s1.size(), s2.size())) {
        throw std::invalid_argument("Cannot find a path of length " + std::to_string(max_length) + " to align given time series.");
    }

    auto accumulated_costs = limited_warping_length_cost(s1, s2, max_length);
    Pair idx_pair = { static_cast<int>(s1.size() - 1), static_cast<int>(s2.size() - 1) };

    int optimal_length = -1;
    double optimal_cost = std::numeric_limits<double>::infinity();

    for (const auto& item : accumulated_costs[idx_pair]) {
        int k = item.first;
        double v = item.second;
        if (v < optimal_cost) {
            optimal_cost = v;
            optimal_length = k;
        }
    }

    auto path = return_path_limited_warping_length(accumulated_costs, idx_pair, optimal_length);
    return { path, std::sqrt(optimal_cost) };
}

PYBIND11_MODULE(ldtw, m) {
    m.def("local_squared_dist", &local_squared_dist, "Compute local squared distance between two time series");
    m.def("max_steps", &max_steps, "Compute maximum number of steps required in a L-DTW process to reach a given cell");
    m.def("limited_warping_length_cost", &limited_warping_length_cost, "Compute accumulated scores necessary for L-DTW");
    m.def("dtw_path_limited_warping_length", &dtw_path_limited_warping_length, "Compute DTW similarity measure between time series under an upper bound constraint on the resulting path length");
}
