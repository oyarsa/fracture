/*
  fracture: Simulation of a scalar fracture model based on Freitas (2007).
  Copyright (C) 2017 Italo Silva

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#define EIGEN_USE_MKL_ALL
#define EIGEN_NO_DEBUG
#include <Eigen/Dense>

using std::size_t;

using time_point = std::chrono::high_resolution_clock::time_point;
using NodeId_t = size_t;
using real = float;
using Amperes = real;
using Ohms = real;

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

bool
almost_equal(real a, real b, real eps = 1e-3)
{
  return std::fabs(a - b) < eps;
}

struct Fuse
{
  const Ohms R;
  const Amperes Imax;

  Fuse(real R, real Imax)
    : R(R)
    , Imax(Imax)
  {
  }

  bool burned(real I) const { return std::fabs(I) > Imax; }
};

struct Edge
{
  const NodeId_t src;
  const NodeId_t dst;
  const Fuse fuse;
  Amperes current;

  Edge(NodeId_t src, NodeId_t dst, Fuse fuse)
    : src(src)
    , dst(dst)
    , fuse(fuse)
    , current(0.0)
  {
  }
};

struct Circuit
{
  std::vector<NodeId_t> nodes;
  std::vector<std::vector<NodeId_t>> levels;
  std::vector<std::vector<std::shared_ptr<Edge>>> adjacencies;
  std::vector<std::vector<std::shared_ptr<Edge>>> inputs;
  Matrix admittance_matrix;
  Matrix coefKCL;

  Circuit(size_t n)
    : nodes(n)
    , levels()
    , adjacencies(n)
    , inputs(n)
    , admittance_matrix(Matrix::Zero(n, n))
    , coefKCL()
  {
  }

  Eigen::Index pseudo_node(NodeId_t v) const { return v - 1; }

  NodeId_t actual_node(size_t i) const { return i + 1; }

  size_t node_count() const { return nodes.size(); }

  size_t effective_node_count() const { return node_count() - 2; }

  void add_node(NodeId_t node, size_t level)
  {
    add_node_to_level(node, level);
    nodes[node] = node;
  }

  void add_node_to_level(NodeId_t node, size_t level)
  {
    while (levels.size() <= level) {
      levels.push_back({});
    }
    levels[level].push_back(node);
  }

  void add_edge(const Edge& edge)
  {
    auto src = edge.src;
    auto dst = edge.dst;
    if (edge.fuse.R != 0)
      admittance_matrix(src, dst) = 1 / edge.fuse.R;

    auto e = std::make_shared<Edge>(edge);
    adjacencies[src].push_back(e);
    inputs[dst].push_back(e);
  }

  void remove_edge(NodeId_t src, NodeId_t dst)
  {
    auto G = admittance_matrix(src, dst);
    admittance_matrix(src, dst) = 0;

    auto u = pseudo_node(src);
    auto v = pseudo_node(dst);

    auto u_in_M = u >= 0 && u < effective_node_count();
    auto v_in_M = v >= 0 && v < effective_node_count();

    if (u_in_M && v_in_M) {
      coefKCL(u, v) = 0;
      coefKCL(v, u) = 0;
    }

    if (u_in_M) {
      coefKCL(u, u) -= G;
      if (almost_equal(coefKCL(u, u), 0.0)) {
        coefKCL(u, u) = 0;
      }
    }

    if (v_in_M) {
      coefKCL(v, v) -= G;
      if (almost_equal(coefKCL(v, v), 0.0)) {
        coefKCL(v, v) = 0;
      }
    }
  }

  void remove_input_edge(const std::shared_ptr<Edge>& edge)
  {
    remove_edge(edge->src, edge->dst);
    auto& ins = inputs[edge->dst];
    ins.erase(std::remove(begin(ins), end(ins), edge), end(ins));
  }

  void remove_node_inputs(NodeId_t node)
  {
    auto& ins = inputs[node];
    for (auto& in : ins) {
      remove_edge(in->src, in->dst);

      auto& adj = adjacencies[in->src];
      adj.erase(std::remove(begin(adj), end(adj), in), end(adj));
      if (adj.empty()) {
        remove_node_inputs(in->src);
      }
    }
    ins.clear();
  }

  NodeId_t source() const { return nodes.front(); }

  NodeId_t sink() const { return nodes.back(); }

  bool connected() const { return adjacencies[source()].size() > 0; }

  bool contains(const std::shared_ptr<Edge>& e) const
  {
    auto& adj = adjacencies[e->src];
    return std::find(begin(adj), end(adj), e) != end(adj);
  }
};

static uint64_t seed_;

void
seed_rand()
{
  auto now = std::chrono::system_clock::now();
  auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
  seed_ = now_ns.time_since_epoch().count();
}

inline uint64_t
nextrand()
{
  // Marsaglia, 2003. Xorshift RNGs, p. 4.
  seed_ ^= seed_ << 13;
  seed_ ^= seed_ >> 7;
  seed_ ^= seed_ << 17;
  return seed_;
}

inline real
nextreal()
{
  return nextrand() / real(UINT64_MAX);
}

constexpr auto k_base = real{ 1 };
constexpr auto base_resistance = Ohms{ k_base };
constexpr auto base_Imax = Amperes{ k_base };

Ohms
random_resist(real D, Ohms R = base_resistance)
{
  if (nextreal() <= D) {
    R *= 0.5 + nextreal();
  }
  return R;
}

Circuit
generate_tilted_circuit(size_t L, real D)
{
  const auto total_nodes = [&] {
    auto total = 2 + (2 * L + 1) * (L / 2);
    const auto LL = (L / 2) * 2;
    const auto diff = L - LL;
    if (diff == 1) {
      total += L;
    }
    return total;
  }();

  auto g = Circuit(total_nodes);
  const NodeId_t source = 0;
  auto level = 0;
  g.add_node(source, level);
  level++;

  NodeId_t next = 1;
  std::vector<NodeId_t> prev(L + 1, source);
  std::vector<NodeId_t> curr(L + 1);
  auto last_count = 1;

  for (auto i = 0u; i < L; i++) {
    // Phase 1
    if (i % 2 == 0) {
      for (auto j = 0u; j < L; j++) {
        const auto x = j;
        const auto y = j + 1;

        const auto node = next++;
        curr[j] = node;
        g.add_node(node, level);

        {
          const auto R = random_resist(D);
          g.add_edge(Edge(prev[x], node, Fuse(R, R)));
        }
        if (y < L + 1 && prev[x] != prev[y]) {
          const auto R = random_resist(D);
          g.add_edge(Edge(prev[y], node, Fuse(R, R)));
        }
      }
      last_count = L;
    }

    // Phase 2
    else if (i % 2 == 1) {
      for (auto j = 0u; j < L + 1; j++) {
        const auto x = int(j) - 1;
        const auto y = j;

        const auto node = next++;
        curr[j] = node;
        g.add_node(node, level);

        if (x >= 0) {
          const auto R = random_resist(D);
          g.add_edge(Edge(prev[x], node, Fuse(R, R)));
        }
        if (y < L) {
          const auto R = random_resist(D);
          g.add_edge(Edge(prev[y], node, Fuse(R, R)));
        }
      }
      last_count = L + 1;
    }

    std::copy_n(begin(curr), last_count, begin(prev));
    level++;
  }

  const auto sink = next;
  g.add_node(sink, level);

  for (auto i = 0u; i < last_count; i++) {
    auto R = random_resist(D);
    g.add_edge(Edge(prev[i], sink, Fuse(R, R)));
  }

  return g;
}

Circuit
generate_hexagon_circuit(size_t L, real D)
{
  const auto n = [&] {
    auto x = 2 + (L / 4) * (4 * L + 2);

    const auto diff = L - (4 * (L / 4));
    if (diff > 0) {
      x += L;
    }
    if (diff > 1) {
      x += L + 1;
    }
    if (diff > 2) {
      x += L + 1;
    }
    return x;
  }();

  auto g = Circuit(n);

  auto next = 0;
  auto level = 0;
  const auto source = next++;

  g.add_node(source, level++);

  std::vector<NodeId_t> prev(L + 1, source);
  std::vector<NodeId_t> curr(L + 1);

  auto last_count = 1;

  for (auto i = 0u; i < L; i++) {
    // Phase 1
    if (i % 4 == 0) {
      for (auto j = 0u; j < L; j++) {
        const auto node = next++;
        curr[j] = node;
        g.add_node(node, level);

        const auto src = prev[j];
        const auto R = random_resist(D);
        g.add_edge(Edge(src, node, Fuse(R, R)));
      }
      last_count = L;
    }

    // Phase 2
    if (i % 4 == 1) {
      for (auto j = 0u; j < L + 1; j++) {
        const auto node = next++;
        curr[j] = node;
        g.add_node(node, level);

        const auto x = j;
        const auto y = int(j) - 1;

        if (x < L) {
          const auto R = random_resist(D);
          g.add_edge(Edge(prev[x], node, Fuse(R, R)));
        }
        if (y >= 0) {
          const auto R = random_resist(D);
          g.add_edge(Edge(prev[y], node, Fuse(R, R)));
        }
      }
      last_count = L + 1;
    }

    // Phase 3
    if (i % 4 == 2) {
      for (auto j = 0u; j < L + 1; j++) {
        const auto node = next++;
        curr[j] = node;
        g.add_node(node, level);

        const auto src = prev[j];
        const auto R = random_resist(D);
        g.add_edge(Edge(src, node, Fuse(R, R)));
      }
      last_count = L + 1;
    }

    // Phase 4
    if (i % 4 == 3) {
      for (auto j = 0u; j < L; j++) {
        const auto node = next++;
        curr[j] = node;
        g.add_node(node, level);

        const auto x = j;
        const auto y = j + 1;

        {
          const auto R = random_resist(D);
          g.add_edge(Edge(prev[x], node, Fuse(R, R)));
        }
        if (y < L + 1) {
          const auto R = random_resist(D);
          g.add_edge(Edge(prev[y], node, Fuse(R, R)));
        }
      }
      last_count = L;
    }

    std::copy_n(begin(curr), last_count, begin(prev));
    level++;
  }

  // Ground
  const auto sink = next++;
  g.add_node(sink, level);
  for (auto i = 0u; i < last_count; i++) {
    const auto R = random_resist(D);
    g.add_edge(Edge(prev[i], sink, Fuse(R, R)));
  }

  return g;
}

Circuit
generate_square_circuit(size_t L, real D)
{
  const auto total_nodes = L * L + 2;
  auto g = Circuit(total_nodes);
  const auto source = NodeId_t{ 0 };
  auto level = 0;
  g.add_node(source, level);
  level++;

  const auto vert = 1.0;
  const auto horiz = 1.0;

  NodeId_t next = 1;
  std::vector<NodeId_t> prev(L, source);
  std::vector<NodeId_t> curr(L);

  for (auto i = 0u; i < L; i++) {
    for (auto j = 0u; j < L; j++) {
      curr[j] = next++;
      g.add_node(curr[j], level);

      const auto hR = random_resist(D);
      g.add_edge(Edge(prev[j], curr[j], Fuse(hR * horiz, hR)));
      if (j > 0) {
        const auto vR = random_resist(D);
        g.add_edge(Edge(curr[j - 1], curr[j], Fuse(vR * vert, vR)));
      }
    }
    std::copy_n(begin(curr), L, begin(prev));
    level++;
  }

  const auto sink = next;
  g.add_node(sink, level);
  for (auto& v : prev) {
    const auto hR = random_resist(D);
    g.add_edge(Edge(v, sink, Fuse(hR * horiz, hR)));
  }

  return g;
}

enum class CircuitType
{
  Square,
  Tilted,
  Hexagon
};

Circuit
generate_circuit(size_t L, real D, CircuitType type)
{
  switch (type) {
    case CircuitType::Square:
      return generate_square_circuit(L, D);
    case CircuitType::Tilted:
      return generate_tilted_circuit(L, D);
    case CircuitType::Hexagon:
      return generate_hexagon_circuit(L, D);
    default:
      return Circuit(L);
  }
}

struct Log
{
  std::vector<std::pair<real, real>> xy;

  void log(real V, real I) { xy.push_back({ V, I }); }

  size_t iterations() const { return xy.size(); }

  void show(std::ostream& out = std::cout) const
  {
    out << "V,I\n";
    for (auto& p : xy) {
      out << p.first << "," << p.second << "\n";
    }
  }
};

std::vector<real>
calculate_ratios(const std::vector<std::shared_ptr<Edge>>& outputs)
{
  std::vector<real> ratios(outputs.size());
  Ohms total(0.0);

  for (auto& e : outputs) {
    total += 1 / e->fuse.R;
  }

  for (auto i = 0u; i < ratios.size(); i++) {
    ratios[i] = (1 / outputs[i]->fuse.R) / total;
  }

  return ratios;
}

Amperes
calculate_current(Circuit& g, real V)
{
  const auto total_current = base_Imax * V;

  {
    auto& level1 = g.adjacencies[0];
    const auto ratios = calculate_ratios(level1);

    for (auto i = 0u; i < level1.size(); i++) {
      level1[i]->current = total_current * ratios[i];
    }
  }

  for (auto lit = begin(g.levels) + 1; lit != end(g.levels); ++lit) {
    for (auto& p : *lit) {
      Amperes current(0.0);
      for (auto& e : g.inputs[p]) {
        current += e->current;
      }

      auto& adj = g.adjacencies[p];
      const auto ratios = calculate_ratios(adj);
      for (auto i = 0u; i < adj.size(); i++) {
        adj[i]->current = current * ratios[i];
      }
    }
  }

  return total_current;
}

void
remove_burned(Circuit& g, real V)
{
  for (const auto p : g.nodes) {
    auto& adj = g.adjacencies[p];
    if (adj.empty())
      continue;

    for (auto it = begin(adj); it != end(adj);) {
      const auto& e = *it;
      if (e->fuse.burned(e->current)) {
        g.remove_input_edge(e);
        it = adj.erase(it);
      } else {
        ++it;
      }
    }

    if (p != g.sink() && adj.empty()) {
      g.remove_node_inputs(p);
    }
  }
}

std::string
flt2str(real f)
{
  std::ostringstream oss;
  oss << std::defaultfloat << f;
  return oss.str();
}

using EdgeMap = std::vector<std::pair<const Edge&, bool>>;

EdgeMap
diff_graph(const Circuit& before, const Circuit& after)
{
  auto m = EdgeMap{};

  for (auto v : before.nodes) {
    for (auto& e : before.adjacencies[v]) {
      m.push_back({ *e, after.contains(e) });
    }
  }

  return m;
}

void
draw_graph(const EdgeMap& em, const std::string& id = "begin")
{
  std::ofstream out{ "graph-" + id + ".dot" };
  out << "digraph {\n";
  out << "  graph [rankdir=LR]\n";
  out << "  node [height=0.05 label=\"\" shape=point width=0.05]\n";
  out << "  edge [arrowsize=0.5 fontsize=8]\n";

  for (const auto& p : em) {
    const auto& e = p.first;
    const auto exists = p.second;
    const auto& f = e.fuse;
    std::string label;
    std::string style;
    std::string head;

    if (!exists) {
      label = "";
      style = "dotted";
      head = "none";
    } else if (f.R == 0) {
      label = "";
      style = "solid";
      head = "none";
    } else {
      label =
        flt2str(f.R) + "Ω " + flt2str(f.Imax) + "A " + flt2str(e.current) + "A";
      style = "solid";
      head = "empty";
    }

    out << "    " << e.src << " -> " << e.dst << "[label=\"" << label
        // << e.src << "->" << e.dst
        << "\" "
        << "style=\"" << style << "\" "
        << "arrowhead=\"" << head << "\" "
        << "]\n";
  }

  out << "}\n";
}

void
verify_kcl(Circuit& g, real Vtotal)
{
  const auto m = g.effective_node_count();

  for (auto i = 0u; i < m; i++) {
    const auto node = g.actual_node(i);

    auto in = 0.0;
    for (auto& e : g.inputs[node])
      in += e->current;

    auto out = 0.0;
    for (auto& e : g.adjacencies[node])
      out += e->current;

    if (!almost_equal(in, out)) {
      std::cout << "Ops\n";
      std::cout << "V: " << Vtotal << "\n";
      std::cout << "Node: " << node << "\n";
      std::cout << "In: " << in << ". Out: " << out << "\n";
    }
  }
}

void
init_kcl(Circuit& g)
{
  const auto m = g.effective_node_count();
  const auto& M = g.admittance_matrix;
  const auto fst = g.actual_node(0);

  //------ Building coefficient matrix
  g.coefKCL = -(M + M.transpose()).block(fst, fst, m, m);
  g.coefKCL.diagonal() =
    (M.rowwise().sum() + M.colwise().sum().transpose()).segment(fst, m);
}

Amperes
calculate_current_kcl(Circuit& g, real Vtotal)
{
  const auto m = g.effective_node_count();
  const auto& M = g.admittance_matrix;
  const auto fst = g.actual_node(0);

  // Coefficient matrix
  const auto& coefKCL = g.coefKCL;

  //------ Building independent term
  Vector currentsKCL =
    (M.topRows(fst) * Vtotal).colwise().sum().segment(fst, m);

  //------ Removing zeroed rows and columns from coef matrix
  const Eigen::Matrix<bool, 1, Eigen::Dynamic> non_zero_cols =
    coefKCL.cast<bool>().colwise().any();

  Matrix A(non_zero_cols.count(), non_zero_cols.count());
  std::vector<size_t> keep;
  keep.reserve(coefKCL.rows());

  for (Eigen::Index u = 0, j = 0; u < coefKCL.cols(); u++) {
    if (!non_zero_cols(u))
      continue;

    for (Eigen::Index v = 0, i = 0; v < coefKCL.rows(); v++) {
      if (non_zero_cols(v)) {
        A(i, j) = coefKCL(v, u);
        i++;
      }
    }
    keep.push_back(u);
    j++;
  }

  //------ Removing elements from the independent term relative to the zeroed rows
  Vector II(keep.size());
  {
    Eigen::Index i = 0;
    for (auto x : keep) {
      II(i++) = currentsKCL(x);
    }
  }

  //------ Solving the system
  Eigen::LLT<Eigen::Ref<Matrix>> solver(A);
  const Vector VV = solver.solve(II);

  //------Going back to a full vector of Voltages, including those
  // nodes that were removed from VV because they weren't connected,
  // and Vcc (with V = Vtotal) and Ground (V = 0).

  // Everyone starts with zero
  Vector V = Vector::Zero(g.node_count());
  // Those that represent Vcc get Vtotal
  V.segment(0, fst) = Vector::Constant(fst, Vtotal);
  {
    // The rest get their voltage respective to the system solution
    Eigen::Index i = 0;
    for (auto x : keep)
      V(g.actual_node(x)) = VV(i++);
  }

  //------Calculating branch currents with Ohms law
  for (auto& adj : g.adjacencies) {
    for (auto& e : adj) {
      e->current = (V[e->src] - V[e->dst]) / e->fuse.R;
    }
  }

  verify_kcl(g, Vtotal);

  //------ Calculating final current
  auto total_current = Amperes(0);
  for (auto& e : g.adjacencies[g.source()])
    total_current += e->current;

  return total_current;
}

Log
simulation(size_t L, real D, CircuitType G, real V0, real deltaV)
{
  auto g = generate_circuit(L, D, G);
  init_kcl(g);

  auto V = V0;
  auto l = Log();

  const auto original = g;
  draw_graph(diff_graph(original, g), "begin");

  while (g.connected()) {
    // auto I = calculate_current(g, V);
    auto I = calculate_current_kcl(g, V);
    l.log(V, I);

    remove_burned(g, V);
    V += deltaV;
  }
  l.log(V, 0);
  draw_graph(diff_graph(original, g), "end");

  return l;
}

time_point
now()
{
  return std::chrono::high_resolution_clock::now();
}

long
elapsed(const time_point& t)
{
  auto e = now() - t;
  return std::chrono::duration_cast<std::chrono::milliseconds>(e).count();
}

int
main(int argc, char** argv)
{
  size_t L = 14;
  real D = 1;
  CircuitType G = CircuitType::Tilted;

  if (argc >= 3) {
    L = std::stoul(argv[1]);
    D = std::stod(argv[2]);
  }
  if (argc >= 4) {
    auto type = std::string{ argv[3] };
    if (type == "s")
      G = CircuitType::Square;
    else if (type == "t")
      G = CircuitType::Tilted;
    else if (type == "h")
      G = CircuitType::Hexagon;
    else {
      std::cout << "Invalid circuit type\n";
      return 1;
    }
  }

  seed_rand();
  auto seed = seed_;
  const auto start = now();
  const auto s = simulation(L, D, G, 0, 0.1);
  const auto& last = s.xy[s.xy.size() - 2];

  printf(
    "G: %c, L: %2zu, D: %2.1f, Iter: %5zu, Time: %4ldms, Vmax: %g, Seed: %zu\n",
    G == CircuitType::Square ? 's' : G == CircuitType::Tilted ? 't' : 'h',
    L,
    D,
    s.iterations(),
    elapsed(start),
    last.first,
    seed);

  std::ofstream out{ "output.csv" };
  s.show(out);
}
