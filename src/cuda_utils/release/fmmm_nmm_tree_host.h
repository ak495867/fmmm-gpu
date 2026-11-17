#pragma once

// Host-side OGDF NewMultipoleMethod reduced quadtree (SubtreeBySubtree).
// Child order LT, RT, LB, RB. Flatten is preorder (matches witnesses_g5).

#include "fmmm_nmm_packing.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <queue>
#include <random>
#include <vector>

namespace lcn::release {
namespace nmm_tree {

constexpr double kMinBoxLength = 1e-300;

struct HostCell {
    double dlc_x = 0.0;
    double dlc_y = 0.0;
    double box_length = 0.0;
    int level = 0;
    int parent = -1;
    int child4[4] = {-1, -1, -1, -1};
    int particle_count = 0;
    double center_x = 0.0;
    double center_y = 0.0;
    bool dead = false;
    std::vector<int> contained;
    std::vector<int> I;
    std::vector<int> D1;
    std::vector<int> D2;
    std::vector<int> M;
    std::vector<std::complex<double>> multipole;
    std::vector<std::complex<double>> local;
};

struct HostTree {
    std::vector<HostCell> cells;
    int root = 0;
};

struct FlatTree {
    int cell_count = 0;
    int precision = 0;
    int expansion_stride = 0;
    std::vector<double> dlc_x;
    std::vector<double> dlc_y;
    std::vector<double> box_length;
    std::vector<double> center_x;
    std::vector<double> center_y;
    std::vector<int> level;
    std::vector<int> parent;
    std::vector<int> child4;
    std::vector<int> particle_count;
    std::vector<int> contained_begin;
    std::vector<int> contained_nodes;
    std::vector<int> I_begin;
    std::vector<int> I_nodes;
    std::vector<int> D1_begin;
    std::vector<int> D1_nodes;
    std::vector<int> D2_begin;
    std::vector<int> D2_nodes;
    std::vector<int> M_begin;
    std::vector<int> M_nodes;
    std::vector<double> multipole_re;
    std::vector<double> multipole_im;
    std::vector<double> local_re;
    std::vector<double> local_im;
};

inline bool nearly_equal_ogdf(double a, double b) {
    const double delta = 1e-10;
    double small_b;
    double big_b;
    if (b > 0) {
        small_b = b * (1.0 - delta);
        big_b = b * (1.0 + delta);
    } else {
        small_b = b * (1.0 + delta);
        big_b = b * (1.0 - delta);
    }
    return small_b <= a && a <= big_b;
}

inline bool is_leaf_cell(const HostCell& cell) {
    return cell.child4[0] < 0 && cell.child4[1] < 0 && cell.child4[2] < 0 && cell.child4[3] < 0;
}

inline int create_child(HostTree& tree, int parent, int which) {
    HostCell& p = tree.cells[parent];
    const double half = p.box_length * 0.5;
    HostCell child;
    child.level = p.level + 1;
    child.parent = parent;
    child.box_length = half;
    // LT, RT, LB, RB
    if (which == 0) {
        child.dlc_x = p.dlc_x;
        child.dlc_y = p.dlc_y + half;
    } else if (which == 1) {
        child.dlc_x = p.dlc_x + half;
        child.dlc_y = p.dlc_y + half;
    } else if (which == 2) {
        child.dlc_x = p.dlc_x;
        child.dlc_y = p.dlc_y;
    } else {
        child.dlc_x = p.dlc_x + half;
        child.dlc_y = p.dlc_y;
    }
    const int idx = static_cast<int>(tree.cells.size());
    tree.cells.push_back(child);
    tree.cells[parent].child4[which] = idx;
    return idx;
}

inline void delete_subtree_mark(HostTree& tree, int idx) {
    if (idx < 0 || tree.cells[idx].dead) {
        return;
    }
    HostCell& cell = tree.cells[idx];
    for (int k = 0; k < 4; ++k) {
        if (cell.child4[k] >= 0) {
            delete_subtree_mark(tree, cell.child4[k]);
        }
        cell.child4[k] = -1;
    }
    cell.dead = true;
    cell.parent = -1;
    cell.particle_count = 0;
    cell.contained.clear();
}

inline void construct_complete_subtree(
        HostTree& tree, int act, int subtree_depth, int act_depth, int act_x, int act_y,
        std::vector<int>& leaf_ptr, int maxindex) {
    if (act_depth < subtree_depth) {
        create_child(tree, act, 0);
        create_child(tree, act, 1);
        create_child(tree, act, 2);
        create_child(tree, act, 3);
        construct_complete_subtree(tree, tree.cells[act].child4[0], subtree_depth, act_depth + 1,
                2 * act_x, 2 * act_y + 1, leaf_ptr, maxindex);
        construct_complete_subtree(tree, tree.cells[act].child4[1], subtree_depth, act_depth + 1,
                2 * act_x + 1, 2 * act_y + 1, leaf_ptr, maxindex);
        construct_complete_subtree(tree, tree.cells[act].child4[2], subtree_depth, act_depth + 1,
                2 * act_x, 2 * act_y, leaf_ptr, maxindex);
        construct_complete_subtree(tree, tree.cells[act].child4[3], subtree_depth, act_depth + 1,
                2 * act_x + 1, 2 * act_y, leaf_ptr, maxindex);
    } else if (act_depth == subtree_depth) {
        leaf_ptr[act_y * maxindex + act_x] = act;
    }
}

inline void set_contained_nodes_for_leaves_grid(HostTree& tree, int subtree_root, const double* px,
        const double* py, const std::vector<int>& leaf_ptr, int maxindex) {
    HostCell& root = tree.cells[subtree_root];
    const double minbox = root.box_length / static_cast<double>(maxindex);
    std::vector<int> particles = std::move(root.contained);
    root.contained.clear();
    for (int v : particles) {
        const double xcoord = px[v] - root.dlc_x;
        const double ycoord = py[v] - root.dlc_y;
        int x_index = static_cast<int>(xcoord / minbox);
        int y_index = static_cast<int>(ycoord / minbox);
        if (x_index < 0) {
            x_index = 0;
        }
        if (y_index < 0) {
            y_index = 0;
        }
        if (x_index >= maxindex) {
            x_index = maxindex - 1;
        }
        if (y_index >= maxindex) {
            y_index = maxindex - 1;
        }
        const int leaf = leaf_ptr[static_cast<size_t>(y_index) * static_cast<size_t>(maxindex) +
                static_cast<size_t>(x_index)];
        tree.cells[leaf].contained.push_back(v);
        tree.cells[leaf].particle_count += 1;
    }
}

inline void set_particlenumber_in_subtree_entries(HostTree& tree, int idx) {
    HostCell& cell = tree.cells[idx];
    if (is_leaf_cell(cell)) {
        return;
    }
    cell.particle_count = 0;
    for (int k = 0; k < 4; ++k) {
        const int c = cell.child4[k];
        if (c < 0) {
            continue;
        }
        set_particlenumber_in_subtree_entries(tree, c);
        cell.particle_count += tree.cells[c].particle_count;
    }
}

inline void delete_empty_subtrees(HostTree& tree, int idx) {
    HostCell& cell = tree.cells[idx];
    for (int k = 0; k < 4; ++k) {
        const int c = cell.child4[k];
        if (c < 0) {
            continue;
        }
        if (tree.cells[c].particle_count == 0) {
            delete_subtree_mark(tree, c);
            cell.child4[k] = -1;
        }
    }
}

inline bool check_and_delete_degenerated_node(HostTree& tree, int& act) {
    HostCell& cell = tree.cells[act];
    int only = -1;
    int nonempty = 0;
    for (int k = 0; k < 4; ++k) {
        if (cell.child4[k] >= 0) {
            ++nonempty;
            only = cell.child4[k];
        }
    }
    if (nonempty != 1) {
        return false;
    }
    HostCell& child = tree.cells[only];
    if (act == tree.root) {
        tree.root = only;
        child.parent = -1;
        cell.dead = true;
        cell.child4[0] = cell.child4[1] = cell.child4[2] = cell.child4[3] = -1;
        act = only;
        return true;
    }
    const int father = cell.parent;
    child.parent = father;
    for (int k = 0; k < 4; ++k) {
        if (tree.cells[father].child4[k] == act) {
            tree.cells[father].child4[k] = only;
            break;
        }
    }
    cell.dead = true;
    cell.parent = -1;
    cell.child4[0] = cell.child4[1] = cell.child4[2] = cell.child4[3] = -1;
    act = only;
    return true;
}

inline void collect_contained_nodes(HostTree& tree, int act, int new_leaf) {
    HostCell& cell = tree.cells[act];
    if (is_leaf_cell(cell)) {
        for (int v : cell.contained) {
            tree.cells[new_leaf].contained.push_back(v);
        }
        cell.contained.clear();
        return;
    }
    for (int k = 0; k < 4; ++k) {
        if (cell.child4[k] >= 0) {
            collect_contained_nodes(tree, cell.child4[k], new_leaf);
        }
    }
}

inline void delete_sparse_subtree(HostTree& tree, int new_leaf) {
    collect_contained_nodes(tree, new_leaf, new_leaf);
    HostCell& cell = tree.cells[new_leaf];
    for (int k = 0; k < 4; ++k) {
        if (cell.child4[k] >= 0) {
            delete_subtree_mark(tree, cell.child4[k]);
            cell.child4[k] = -1;
        }
    }
}

inline bool quad_helper(double minx, double miny, double maxx, double maxy, double blx, double bly,
        double trx, double try_, double dlc_x, double dlc_y) {
    blx += dlc_x;
    bly += dlc_y;
    trx += dlc_x;
    try_ += dlc_y;
    const bool inside = (blx <= minx && maxx < trx && bly <= miny && maxy < try_);
    const bool point_eq = (minx == maxx && miny == maxy && maxx == trx && maxy == try_ &&
            trx == blx && try_ == bly);
    return inside || point_eq;
}

inline bool in_lt_quad(const HostCell& cell, double minx, double miny, double maxx, double maxy) {
    const double lo = cell.box_length * 0.5;
    const double hi = cell.box_length;
    return quad_helper(minx, miny, maxx, maxy, 0.0, lo, lo, hi, cell.dlc_x, cell.dlc_y);
}

inline bool in_rt_quad(const HostCell& cell, double minx, double miny, double maxx, double maxy) {
    const double lo = cell.box_length * 0.5;
    const double hi = cell.box_length;
    return quad_helper(minx, miny, maxx, maxy, lo, lo, hi, hi, cell.dlc_x, cell.dlc_y);
}

inline bool in_lb_quad(const HostCell& cell, double minx, double miny, double maxx, double maxy) {
    const double lo = cell.box_length * 0.5;
    return quad_helper(minx, miny, maxx, maxy, 0.0, 0.0, lo, lo, cell.dlc_x, cell.dlc_y);
}

inline bool in_rb_quad(const HostCell& cell, double minx, double miny, double maxx, double maxy) {
    const double lo = cell.box_length * 0.5;
    const double hi = cell.box_length;
    return quad_helper(minx, miny, maxx, maxy, lo, 0.0, hi, lo, cell.dlc_x, cell.dlc_y);
}

inline void find_small_cell_iteratively(
        HostCell& cell, double minx, double miny, double maxx, double maxy) {
    while ((maxx - minx) >= kMinBoxLength || (maxy - miny) >= kMinBoxLength) {
        const double new_box = cell.box_length * 0.5;
        double new_dlc_x = cell.dlc_x;
        double new_dlc_y = cell.dlc_y;
        if (in_lt_quad(cell, minx, miny, maxx, maxy)) {
            new_dlc_y += new_box;
        } else if (in_rt_quad(cell, minx, miny, maxx, maxy)) {
            new_dlc_x += new_box;
            new_dlc_y += new_box;
        } else if (in_lb_quad(cell, minx, miny, maxx, maxy)) {
            // keep
        } else if (in_rb_quad(cell, minx, miny, maxx, maxy)) {
            new_dlc_x += new_box;
        } else {
            return;
        }
        cell.level += 1;
        cell.box_length = new_box;
        cell.dlc_x = new_dlc_x;
        cell.dlc_y = new_dlc_y;
    }
}

inline bool find_smallest_quad(HostTree& tree, int act, const double* px, const double* py) {
    HostCell& cell = tree.cells[act];
    if (cell.contained.empty()) {
        return false;
    }
    double minx = px[cell.contained.front()];
    double maxx = minx;
    double miny = py[cell.contained.front()];
    double maxy = miny;
    for (int v : cell.contained) {
        minx = std::min(minx, px[v]);
        maxx = std::max(maxx, px[v]);
        miny = std::min(miny, py[v]);
        maxy = std::max(maxy, py[v]);
    }
    if (minx == maxx && miny == maxy) {
        return false;
    }
    find_small_cell_iteratively(cell, minx, miny, maxx, maxy);
    return true;
}

inline void construct_reduced_subtree(
        HostTree& tree, int act, const double* px, const double* py, int particles_in_leaf,
        std::vector<int>& new_subtree_roots);

inline void construct_subtree(HostTree& tree, int subtree_root, const double* px, const double* py,
        int particles_in_leaf, std::vector<int>& new_subtree_roots) {
    const int n = tree.cells[subtree_root].particle_count;
    const int subtree_depth = static_cast<int>(std::max(1.0, std::floor(std::log(static_cast<double>(n)) / std::log(4.0)) - 2.0));
    int maxindex = 1;
    for (int i = 1; i <= subtree_depth; ++i) {
        maxindex *= 2;
    }
    const double subtree_min_box = tree.cells[subtree_root].box_length / static_cast<double>(maxindex);
    if (subtree_min_box < kMinBoxLength) {
        return;
    }
    if (!find_smallest_quad(tree, subtree_root, px, py)) {
        return;
    }
    std::vector<int> leaf_ptr(static_cast<size_t>(maxindex) * static_cast<size_t>(maxindex), -1);
    construct_complete_subtree(tree, subtree_root, subtree_depth, 0, 0, 0, leaf_ptr, maxindex);
    set_contained_nodes_for_leaves_grid(tree, subtree_root, px, py, leaf_ptr, maxindex);
    set_particlenumber_in_subtree_entries(tree, subtree_root);
    construct_reduced_subtree(tree, subtree_root, px, py, particles_in_leaf, new_subtree_roots);
}

inline void construct_reduced_subtree(
        HostTree& tree, int act, const double* px, const double* py, int particles_in_leaf,
        std::vector<int>& new_subtree_roots) {
    do {
        delete_empty_subtrees(tree, act);
    } while (check_and_delete_degenerated_node(tree, act));

    HostCell& cell = tree.cells[act];
    if (!is_leaf_cell(cell) && cell.particle_count <= particles_in_leaf) {
        delete_sparse_subtree(tree, act);
    }

    if (is_leaf_cell(tree.cells[act]) && tree.cells[act].particle_count > particles_in_leaf) {
        new_subtree_roots.push_back(act);
    } else if (is_leaf_cell(tree.cells[act]) &&
            tree.cells[act].particle_count <= particles_in_leaf) {
        find_smallest_quad(tree, act, px, py);
    } else if (!is_leaf_cell(tree.cells[act])) {
        for (int k = 0; k < 4; ++k) {
            const int c = tree.cells[act].child4[k];
            if (c >= 0) {
                construct_reduced_subtree(tree, c, px, py, particles_in_leaf, new_subtree_roots);
            }
        }
    }
}

inline void build_up_red_quad_tree_subtree_by_subtree(
        HostTree& tree, const double* px, const double* py, int n, int particles_in_leaf) {
    std::vector<int> act_roots;
    std::vector<int> new_roots;
    act_roots.push_back(tree.root);
    while (!act_roots.empty()) {
        new_roots.clear();
        for (int r : act_roots) {
            construct_subtree(tree, r, px, py, particles_in_leaf, new_roots);
        }
        act_roots.swap(new_roots);
    }
    (void)n;
}

inline long long binko(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    long long r = 1;
    for (int i = 1; i <= k; ++i) {
        r = r * (n - k + i) / i;
    }
    return r;
}

inline void set_center(HostCell& cell, std::mt19937& rng) {
    constexpr int kBillion = 1000000000;
    double cx = cell.dlc_x + cell.box_length * 0.5;
    double cy = cell.dlc_y + cell.box_length * 0.5;
    std::uniform_int_distribution<int> dist(1, kBillion);
    const double rand_y = double(dist(rng) + 1) / (kBillion + 2);
    cy += 0.001 * cell.box_length * rand_y;
    cell.center_x = cx;
    cell.center_y = cy;
}

inline void form_multipole_leaf(HostCell& cell, const double* px, const double* py, int precision) {
    cell.multipole.assign(precision + 1, std::complex<double>(0, 0));
    cell.multipole[0] = std::complex<double>(static_cast<double>(cell.contained.size()), 0);
    const std::complex<double> z0(cell.center_x, cell.center_y);
    for (int v : cell.contained) {
        std::complex<double> zv(px[v], py[v]);
        std::complex<double> z = zv - z0;
        std::complex<double> pow = z;
        for (int k = 1; k <= precision; ++k) {
            cell.multipole[k] += (-pow) / static_cast<double>(k);
            pow *= z;
        }
    }
}

inline void shift_multipole_to_parent(HostCell& child, HostCell& father, int precision) {
    const std::complex<double> z0(child.center_x, child.center_y);
    const std::complex<double> z1(father.center_x, father.center_y);
    std::vector<std::complex<double>> zdiff(precision + 1);
    zdiff[0] = 1.0;
    for (int i = 1; i <= precision; ++i) {
        zdiff[i] = zdiff[i - 1] * (z0 - z1);
    }
    father.multipole[0] += child.multipole[0];
    for (int k = 1; k <= precision; ++k) {
        std::complex<double> sum = (child.multipole[0] * (-zdiff[k])) / static_cast<double>(k);
        for (int s = 1; s <= k; ++s) {
            sum += child.multipole[s] * zdiff[k - s] * static_cast<double>(binko(k - 1, s - 1));
        }
        father.multipole[k] += sum;
    }
}

inline void form_multipoles(HostTree& tree, int idx, const double* px, const double* py, int precision,
        std::mt19937& rng, std::vector<int>& leaves) {
    HostCell& cell = tree.cells[idx];
    cell.multipole.assign(precision + 1, std::complex<double>(0, 0));
    cell.local.assign(precision + 1, std::complex<double>(0, 0));
    set_center(cell, rng);
    if (is_leaf_cell(cell)) {
        leaves.push_back(idx);
        form_multipole_leaf(cell, px, py, precision);
        return;
    }
    for (int k = 0; k < 4; ++k) {
        const int c = cell.child4[k];
        if (c >= 0) {
            form_multipoles(tree, c, px, py, precision, rng, leaves);
            shift_multipole_to_parent(tree.cells[c], cell, precision);
        }
    }
}

inline bool well_separated(const HostCell& a, const HostCell& b) {
    const double box1 = a.box_length;
    const double box2 = b.box_length;
    double x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max;
    if (box1 <= box2) {
        x1_min = a.dlc_x;
        x1_max = a.dlc_x + box1;
        y1_min = a.dlc_y;
        y1_max = a.dlc_y + box1;
        x2_min = b.dlc_x - box2;
        x2_max = b.dlc_x + 2 * box2;
        y2_min = b.dlc_y - box2;
        y2_max = b.dlc_y + 2 * box2;
    } else {
        x1_min = a.dlc_x - box1;
        x1_max = a.dlc_x + 2 * box1;
        y1_min = a.dlc_y - box1;
        y1_max = a.dlc_y + 2 * box1;
        x2_min = b.dlc_x;
        x2_max = b.dlc_x + box2;
        y2_min = b.dlc_y;
        y2_max = b.dlc_y + box2;
    }
    const bool x_overlap = !(x1_max <= x2_min || nearly_equal_ogdf(x1_max, x2_min) ||
            x2_max <= x1_min || nearly_equal_ogdf(x2_max, x1_min));
    const bool y_overlap = !(y1_max <= y2_min || nearly_equal_ogdf(y1_max, y2_min) ||
            y2_max <= y1_min || nearly_equal_ogdf(y2_max, y1_min));
    return !(x_overlap && y_overlap);
}

inline bool bordering(const HostCell& a, const HostCell& b) {
    const double box1 = a.box_length;
    const double box2 = b.box_length;
    double x1_min = a.dlc_x;
    double x1_max = a.dlc_x + box1;
    double y1_min = a.dlc_y;
    double y1_max = a.dlc_y + box1;
    double x2_min = b.dlc_x;
    double x2_max = b.dlc_x + box2;
    double y2_min = b.dlc_y;
    double y2_max = b.dlc_y + box2;

    auto leq = [](double p, double q) { return p <= q || nearly_equal_ogdf(p, q); };
    auto box_contained = [&](double x1min, double x2min, double x1max, double x2max, double y1min,
                                  double y2min, double y1max, double y2max) {
        return (leq(x2min, x1min) && leq(x1max, x2max) && leq(y2min, y1min) && leq(y1max, y2max)) ||
                (leq(x1min, x2min) && leq(x2max, x1max) && leq(y1min, y2min) && leq(y2max, y1max));
    };
    auto shift_box = [](double& x1min, double& x1max, double& x2min, double& x2max, double& y1min,
                             double& y1max, double& y2min, double& y2max, double length) {
        if (x1min < x2min) {
            x1min += length;
            x1max += length;
        } else if (x1max > x2max) {
            x1min -= length;
            x1max -= length;
        }
        if (y1min < y2min) {
            y1min += length;
            y1max += length;
        } else if (y1max > y2max) {
            y1min -= length;
            y1max -= length;
        }
    };

    if (box_contained(x1_min, x2_min, x1_max, x2_max, y1_min, y2_min, y1_max, y2_max)) {
        return false;
    }
    if (box1 <= box2) {
        shift_box(x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max, box1);
    } else {
        shift_box(x2_min, x2_max, x1_min, x1_max, y2_min, y2_max, y1_min, y1_max, box2);
    }
    return box_contained(x1_min, x2_min, x1_max, x2_max, y1_min, y2_min, y1_max, y2_max);
}

// OGDF add_local_expansion(ptr_0 multipole -> ptr_1 local)
inline void add_local_from_multipole(HostCell& target, const HostCell& source, int precision) {
    const std::complex<double> z0(source.center_x, source.center_y);
    const std::complex<double> z1(target.center_x, target.center_y);
    const std::complex<double> z = z1 - z0;
    if (std::abs(z) < 1e-300) {
        return;
    }
    if (target.local.size() < static_cast<size_t>(precision + 1)) {
        target.local.assign(precision + 1, std::complex<double>(0, 0));
    }

    std::complex<double> sum = source.multipole[0] * std::log(z);
    std::complex<double> z_over_k = z;
    for (int k = 1; k <= precision; ++k) {
        sum += source.multipole[k] / z_over_k;
        z_over_k *= z;
    }
    target.local[0] += sum;

    std::complex<double> z_over_s = z;
    for (int s = 1; s <= precision; ++s) {
        const std::complex<double> pow_m1_s_plus_1 = (((s + 1) % 2 == 0) ? 1.0 : -1.0);
        const std::complex<double> pow_m1_s = ((pow_m1_s_plus_1 == 1.0) ? -1.0 : 1.0);
        sum = pow_m1_s_plus_1 * source.multipole[0] / (z_over_s * static_cast<double>(s));
        const std::complex<double> factor = pow_m1_s / z_over_s;
        z_over_s *= z;
        std::complex<double> sum2(0, 0);
        std::complex<double> z_over_k2 = z;
        for (int k = 1; k <= precision; ++k) {
            sum2 += static_cast<double>(binko(s + k - 1, k - 1)) * source.multipole[k] / z_over_k2;
            z_over_k2 *= z;
        }
        target.local[s] += sum + factor * sum2;
    }
}

inline void add_shifted_local_exp_of_parent(HostCell& node, const HostCell& father, int precision) {
    const std::complex<double> z0(father.center_x, father.center_y);
    const std::complex<double> z1(node.center_x, node.center_y);
    std::vector<std::complex<double>> zdiff(precision + 1);
    zdiff[0] = 1.0;
    for (int i = 1; i <= precision; ++i) {
        zdiff[i] = zdiff[i - 1] * (z1 - z0);
    }
    if (node.local.size() < static_cast<size_t>(precision + 1)) {
        node.local.assign(precision + 1, std::complex<double>(0, 0));
    }
    for (int k = 0; k <= precision; ++k) {
        std::complex<double> sum(0, 0);
        for (int n = k; n <= precision; ++n) {
            sum += static_cast<double>(binko(n, k)) * father.local[n] * zdiff[n - k];
        }
        node.local[k] += sum;
    }
}

inline void add_local_expansion_of_leaf(
        HostCell& target, const HostCell& leaf, const double* px, const double* py, int precision) {
    const std::complex<double> z1(target.center_x, target.center_y);
    if (target.local.size() < static_cast<size_t>(precision + 1)) {
        target.local.assign(precision + 1, std::complex<double>(0, 0));
    }
    for (int v : leaf.contained) {
        const std::complex<double> z0(px[v], py[v]);
        target.local[0] += std::log(z1 - z0);
        std::complex<double> z_over_s = z1 - z0;
        for (int s = 1; s <= precision; ++s) {
            const std::complex<double> pow_m1 = (((s + 1) % 2 == 0) ? 1.0 : -1.0);
            target.local[s] += pow_m1 / (z_over_s * static_cast<double>(s));
            z_over_s *= (z1 - z0);
        }
    }
}

inline void compute_wsprls(HostTree& tree, int idx, const double* px, const double* py, int precision) {
    HostCell& act = tree.cells[idx];
    act.I.clear();
    act.D1.clear();
    act.D2.clear();
    act.M.clear();
    if (act.local.size() < static_cast<size_t>(precision + 1)) {
        act.local.assign(precision + 1, std::complex<double>(0, 0));
    }

    std::queue<int> E;
    std::vector<int> L;
    std::vector<int> L2;
    std::vector<int> I;
    std::vector<int> D1;
    std::vector<int> D2;

    if (act.parent < 0) {
        for (int k = 0; k < 4; ++k) {
            if (act.child4[k] >= 0) {
                E.push(act.child4[k]);
            }
        }
    } else {
        const HostCell& father = tree.cells[act.parent];
        for (int v : father.D1) {
            E.push(v);
        }
        for (int v : father.I) {
            E.push(v);
        }
    }

    while (!E.empty()) {
        const int selected = E.front();
        E.pop();
        const HostCell& sel = tree.cells[selected];
        if (well_separated(act, sel)) {
            L.push_back(selected);
        } else if (act.level < sel.level) {
            I.push_back(selected);
        } else if (!is_leaf_cell(sel)) {
            for (int k = 0; k < 4; ++k) {
                if (sel.child4[k] >= 0) {
                    E.push(sel.child4[k]);
                }
            }
        } else if (bordering(act, sel)) {
            D1.push_back(selected);
        } else if (selected != idx && is_leaf_cell(act)) {
            D2.push_back(selected);
        } else if (selected != idx) {
            L2.push_back(selected);
        }
    }
    act.I = I;
    act.D1 = D1;
    act.D2 = D2;

    if (act.parent >= 0) {
        add_shifted_local_exp_of_parent(act, tree.cells[act.parent], precision);
    }
    for (int s : L) {
        add_local_from_multipole(act, tree.cells[s], precision);
    }
    for (int s : L2) {
        add_local_expansion_of_leaf(act, tree.cells[s], px, py, precision);
    }

    if (!is_leaf_cell(act)) {
        for (int k = 0; k < 4; ++k) {
            if (act.child4[k] >= 0) {
                compute_wsprls(tree, act.child4[k], px, py, precision);
            }
        }
    } else {
        std::queue<int> Iwork;
        for (int v : act.I) {
            Iwork.push(v);
        }
        act.I.clear();
        std::vector<int> M;
        D1 = act.D1;
        D2 = act.D2;
        while (!Iwork.empty()) {
            const int selected = Iwork.front();
            Iwork.pop();
            const HostCell& sel = tree.cells[selected];
            if (is_leaf_cell(sel)) {
                if (bordering(act, sel)) {
                    D1.push_back(selected);
                } else {
                    D2.push_back(selected);
                }
            } else if (bordering(act, sel)) {
                for (int k = 0; k < 4; ++k) {
                    if (sel.child4[k] >= 0) {
                        Iwork.push(sel.child4[k]);
                    }
                }
            } else {
                M.push_back(selected);
            }
        }
        act.D1 = D1;
        act.D2 = D2;
        act.M = M;
    }
}

inline HostTree build_nmm_tree_host(
        const double* px, const double* py, int n, const FmmmNmmOptions& options) {
    HostTree tree;
    HostCell root;
    root.dlc_x = options.down_left_x;
    root.dlc_y = options.down_left_y;
    root.box_length = options.box_length;
    root.level = 0;
    root.parent = -1;
    root.particle_count = n;
    root.contained.resize(n);
    for (int i = 0; i < n; ++i) {
        root.contained[i] = i;
    }
    tree.cells.push_back(root);
    tree.root = 0;

    const int pil = options.particles_in_leaf > 0 ? options.particles_in_leaf : 25;
    build_up_red_quad_tree_subtree_by_subtree(tree, px, py, n, pil);

    // Sync particle_count on leaves from contained size after reductions.
    for (auto& cell : tree.cells) {
        if (!cell.dead && is_leaf_cell(cell)) {
            cell.particle_count = static_cast<int>(cell.contained.size());
        }
    }
    if (!tree.cells[tree.root].dead) {
        set_particlenumber_in_subtree_entries(tree, tree.root);
    }

    std::mt19937 rng(static_cast<std::uint32_t>(options.seed ? options.seed : 1));
    const int precision = options.precision > 0 ? options.precision : 4;
    std::vector<int> leaves;
    form_multipoles(tree, tree.root, px, py, precision, rng, leaves);
    compute_wsprls(tree, tree.root, px, py, precision);
    return tree;
}

inline void preorder_collect(const HostTree& tree, int idx, std::vector<int>& order) {
    if (idx < 0 || tree.cells[idx].dead) {
        return;
    }
    order.push_back(idx);
    for (int k = 0; k < 4; ++k) {
        preorder_collect(tree, tree.cells[idx].child4[k], order);
    }
}

inline FlatTree flatten_tree(const HostTree& tree, int precision) {
    std::vector<int> order;
    preorder_collect(tree, tree.root, order);
    std::vector<int> remap(tree.cells.size(), -1);
    for (size_t i = 0; i < order.size(); ++i) {
        remap[order[i]] = static_cast<int>(i);
    }

    FlatTree flat;
    flat.cell_count = static_cast<int>(order.size());
    flat.precision = precision;
    flat.expansion_stride = precision + 1;
    flat.dlc_x.resize(flat.cell_count);
    flat.dlc_y.resize(flat.cell_count);
    flat.box_length.resize(flat.cell_count);
    flat.center_x.resize(flat.cell_count);
    flat.center_y.resize(flat.cell_count);
    flat.level.resize(flat.cell_count);
    flat.parent.resize(flat.cell_count);
    flat.child4.resize(flat.cell_count * 4);
    flat.particle_count.resize(flat.cell_count);
    flat.contained_begin = {0};
    flat.I_begin = {0};
    flat.D1_begin = {0};
    flat.D2_begin = {0};
    flat.M_begin = {0};
    flat.multipole_re.assign(flat.cell_count * flat.expansion_stride, 0.0);
    flat.multipole_im.assign(flat.cell_count * flat.expansion_stride, 0.0);
    flat.local_re.assign(flat.cell_count * flat.expansion_stride, 0.0);
    flat.local_im.assign(flat.cell_count * flat.expansion_stride, 0.0);

    auto map_list = [&](const std::vector<int>& src, std::vector<int>& dst) {
        for (int v : src) {
            if (v >= 0 && v < static_cast<int>(remap.size()) && remap[v] >= 0) {
                dst.push_back(remap[v]);
            }
        }
    };

    for (int i = 0; i < flat.cell_count; ++i) {
        const HostCell& c = tree.cells[order[i]];
        flat.dlc_x[i] = c.dlc_x;
        flat.dlc_y[i] = c.dlc_y;
        flat.box_length[i] = c.box_length;
        flat.center_x[i] = c.center_x;
        flat.center_y[i] = c.center_y;
        flat.level[i] = c.level;
        flat.parent[i] = (c.parent >= 0) ? remap[c.parent] : -1;
        for (int k = 0; k < 4; ++k) {
            flat.child4[i * 4 + k] = (c.child4[k] >= 0) ? remap[c.child4[k]] : -1;
        }
        flat.particle_count[i] = c.particle_count;
        for (int v : c.contained) {
            flat.contained_nodes.push_back(v);
        }
        flat.contained_begin.push_back(static_cast<int>(flat.contained_nodes.size()));
        map_list(c.I, flat.I_nodes);
        flat.I_begin.push_back(static_cast<int>(flat.I_nodes.size()));
        map_list(c.D1, flat.D1_nodes);
        flat.D1_begin.push_back(static_cast<int>(flat.D1_nodes.size()));
        map_list(c.D2, flat.D2_nodes);
        flat.D2_begin.push_back(static_cast<int>(flat.D2_nodes.size()));
        map_list(c.M, flat.M_nodes);
        flat.M_begin.push_back(static_cast<int>(flat.M_nodes.size()));
        for (int k = 0; k <= precision; ++k) {
            const auto mp = (k < static_cast<int>(c.multipole.size())) ? c.multipole[k]
                                                                       : std::complex<double>(0, 0);
            const auto loc =
                    (k < static_cast<int>(c.local.size())) ? c.local[k] : std::complex<double>(0, 0);
            flat.multipole_re[i * flat.expansion_stride + k] = mp.real();
            flat.multipole_im[i * flat.expansion_stride + k] = mp.imag();
            flat.local_re[i * flat.expansion_stride + k] = loc.real();
            flat.local_im[i * flat.expansion_stride + k] = loc.imag();
        }
    }
    return flat;
}

} // namespace nmm_tree
} // namespace lcn::release
