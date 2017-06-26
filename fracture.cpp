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
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#define EIGEN_NO_DEBUG

using std::size_t;

using time_point = std::chrono::high_resolution_clock::time_point;
using NodeId_t = size_t;
using real = double;
using Amperes = real;
using Ohms = real;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

const real inf = 1e9;

struct Fuse
{
  Ohms R;
  Amperes Imax;

  Fuse(real R, real Imax)
    : R(R)
    , Imax(Imax)
  {
  }

  bool blown(real I) { return I > Imax; }
};

struct Edge
{
  NodeId_t src;
  NodeId_t dst;
  Fuse fuse;
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
  std::vector<std::size_t> node_level;
  std::vector<std::vector<NodeId_t>> levels;
  std::vector<std::vector<std::shared_ptr<Edge>>> adjacencies;
  std::vector<std::vector<std::shared_ptr<Edge>>> inputs;
  Matrix admittance_matrix;

  Circuit(size_t n)
    : nodes(n)
    , node_level(n)
    , levels()
    , adjacencies(n)
    , inputs(n)
    , admittance_matrix(Matrix::Zero(n, n))
  {
  }

  std::size_t pseudo_node(NodeId_t v) const
  {
    return v - 1 - levels.at(1).size();
  }

  NodeId_t actual_node(std::size_t i) const
  {
    return i + 1 + levels.at(1).size();
  }

  std::size_t effective_node_count() const
  {
    return nodes.size() - 2 -
           (levels.at(1).size() + levels.at(levels.size() - 2).size());
  }

  void add_node(NodeId_t node, size_t level)
  {
    add_node_to_level(node, level);
    node_level[node] = level;
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
    adjacencies[edge.src].push_back(e);
    inputs[edge.dst].push_back(e);
  }

  void remove_input_edge(const std::shared_ptr<Edge>& edge)
  {
    admittance_matrix(edge->src, edge->dst) = 0;
    auto& ins = inputs[edge->dst];
    ins.erase(std::remove(begin(ins), end(ins), edge), end(ins));
  }

  void remove_node_inputs(NodeId_t node)
  {
    auto& ins = inputs[node];
    for (auto& in : ins) {
      auto& adj = adjacencies[in->src];
      admittance_matrix(in->src, in->dst) = 0;
      adj.erase(std::remove(begin(adj), end(adj), in), end(adj));
      if (adj.empty()) {
        remove_node_inputs(in->src);
      }
    }
    ins.clear();
  }

  NodeId_t source() const { return nodes.front(); }

  NodeId_t sink() const { return nodes.back(); }

  bool connected() { return adjacencies[source()].size() > 0; }

  bool contains(const std::shared_ptr<Edge>& e) const
  {
    auto& adj = adjacencies[e->src];
    return std::find(begin(adj), end(adj), e) != end(adj);
  }

  Ohms edge_between(NodeId_t u, NodeId_t v) const
  {
    // auto R = admittance_matrix(u, v);
    // if (R)
    //   return R;
    // return admittance_matrix(v, u);
    auto e = find_edge(u, v);
    if (e)
      return e->fuse.R;
    e = find_edge(v, u);
    if (e)
      return e->fuse.R;
    return 0;
  }

  const Edge* find_edge(NodeId_t src, NodeId_t dst) const
  {
    auto& adj = adjacencies[src];
    auto it = std::find_if(
      begin(adj), end(adj), [&dst](auto& e) { return e->dst == dst; });
    if (it != end(adj))
      return it->get();
    else
      return nullptr;
  }
};

static uint64_t seed_;

void
seed_rand()
{
  seed_ = std::time(0);
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

auto base_resistance = Ohms{ 20 };
auto base_Imax = Amperes{ 20 };

Ohms
random_resist(real D, Ohms R = base_resistance)
{
  if (nextreal() <= D) {
    R *= 0.5 + nextreal();
  }
  return R;
}

Circuit
generate_titled_circuit(size_t L, real D)
{
  auto total_nodes = 2 + (L / 2) * (L + L + 1) + L * (L % 2);
  auto g = Circuit(total_nodes);
  NodeId_t source = 0;
  auto level = 0;
  g.add_node(source, level);
  level++;

  NodeId_t next = 1;
  std::vector<NodeId_t> prev(L + 1);
  std::vector<NodeId_t> curr(L + 1);

  for (auto i = 0u; i < L; i++) {
    prev[i] = next++;
    g.add_node(prev[i], level);
    g.add_edge(Edge(source, prev[i], Fuse(0, inf)));
  }
  level++;

  for (auto i = 1u; i < L; i++) {
    auto n = L + (i % 2);
    int max = L + (i % 2 ? 0 : 1);

    for (auto j = 0u; j < n; j++) {
      int nx, ny;
      if (i % 2) {
        nx = j - 1;
        ny = j;
      } else {
        nx = j;
        ny = j + 1;
      }

      curr[j] = next++;
      g.add_node(curr[j], level);
      if (nx >= 0) {
        auto R = random_resist(D);
        g.add_edge(Edge(prev[nx], curr[j], Fuse(R, R)));
      }
      if (ny < max) {
        auto R = random_resist(D);
        g.add_edge(Edge(prev[ny], curr[j], Fuse(R, R)));
      }
    }

    for (auto j = 0u; j < n; j++) {
      prev[j] = curr[j];
    }
    level++;
  }

  NodeId_t sink = next;
  g.add_node(sink, level);
  for (auto i = 0u; i < L + (L - 1) % 2; i++) {
    g.add_edge(Edge(prev[i], sink, Fuse(0, inf)));
  }

  return g;
}

Circuit
generate_hexagon_circuit(size_t L, real D)
{
  return Circuit(L);
}

Circuit
generate_square_circuit(size_t L, real D)
{
  auto total_nodes = L * L + 2;
  auto g = Circuit(total_nodes);
  auto source = NodeId_t{ 0 };
  auto level = 0;
  g.add_node(source, level);
  level++;

  NodeId_t next = 1;
  std::vector<NodeId_t> prev(L);
  std::vector<NodeId_t> curr(L);

  for (auto i = 0u; i < L; i++) {
    prev[i] = next++;
    g.add_node(prev[i], level);
    g.add_edge(Edge(source, prev[i], Fuse(0, inf)));
  }
  level++;

  auto vert = 0.99;
  auto horiz = 0.01;

  for (auto i = 1u; i < L; i++) {
    for (auto j = 0u; j < L; j++) {
      curr[j] = next++;
      g.add_node(curr[j], level);

      auto hR = random_resist(D);
      g.add_edge(Edge(prev[j], curr[j], Fuse(hR * horiz, hR)));
      if (j > 0) {
        auto vR = random_resist(D);
        g.add_edge(Edge(curr[j - 1], curr[j], Fuse(vR * vert, vR)));
      }
    }
    for (auto j = 0u; j < L; j++) {
      prev[j] = curr[j];
    }
    level++;
  }

  auto sink = next;
  g.add_node(sink, level);
  for (auto& v : prev) {
    g.add_edge(Edge(v, sink, Fuse(0, inf)));
  }

  return g;
}

enum class CircuitType
{
  Square,
  Tilted,
  Hexagon
};
auto CurrentType = CircuitType::Tilted;

Circuit
generate_circuit(size_t L, real D)
{
  switch (CurrentType) {
    case CircuitType::Square:
      return generate_square_circuit(L, D);
    case CircuitType::Tilted:
      return generate_titled_circuit(L, D);
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

  void show(std::ostream& out = std::cout)
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
  if (ratios.size() == 1) {
    ratios[0] = 1;
    return ratios;
  }

  Ohms total(0.0);
  for (auto& e : outputs) {
    total += e->fuse.R;
  }

  for (auto i = 0u; i < ratios.size(); i++) {
    ratios[i] = (total - outputs[i]->fuse.R) / total;
  }

  return ratios;
}

Amperes
calculate_current(Circuit& g, real V)
{
  // Total current in circuit. Maybe improve this?
  const auto total_current = base_Imax * V;

  // Distribute the current evenly to the first level.
  for (auto& out : g.adjacencies[0]) {
    out->current = total_current / g.adjacencies[0].size();
  }

  for (auto lit = begin(g.levels) + 1; lit != end(g.levels); ++lit) {
    for (auto& p : *lit) {
      Amperes current(0.0);
      for (auto& e : g.inputs[p]) {
        current += e->current;
      }

      auto& adj = g.adjacencies[p];
      auto ratios = calculate_ratios(adj);
      for (auto i = 0u; i < adj.size(); i++) {
        adj[i]->current = current * ratios[i];
      }
    }
  }

  return total_current;
}

void
iteration(Circuit& g, real V)
{
  for (auto p : g.nodes) {
    auto& adj = g.adjacencies[p];
    if (adj.empty())
      continue;
    auto it = begin(adj);

    while (it != end(adj)) {
      auto& e = *it;
      if (e->fuse.blown(e->current)) {
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

  for (auto& p : em) {
    auto& e = p.first;
    auto exists = p.second;
    auto& f = e.fuse;
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

  // std::cout << "Graph printed: " << id << "\n";
}

void
print_la(const std::pair<Matrix, Vector>& Ab)
{
  const auto& A = Ab.first;
  const auto& b = Ab.second;
  const auto m = A.rows();
  const auto width = 9;
  const auto precision = 4;

  for (auto i = 0u; i < m; i++) {
    for (auto j = 0u; j < m + 2; j++) {
      std::cout << std::setw(width) << std::setprecision(precision);
      if (j < m) {
        std::cout << A(i, j);
      } else if (j == m) {
        std::cout << "X" << (i + 1);
      } else if (j > m) {
        std::cout << b(i);
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

bool
almost_equal(real a, real b, real eps = 1e-9)
{
  return std::fabs(a - b) < eps;
}

void
verify_kcl(Circuit& g)
{
  auto m = g.effective_node_count();

  for (auto i = 0u; i < m; i++) {
    auto node = g.actual_node(i);

    auto in = 0.0;
    for (auto& e : g.inputs[node])
      in += e->current;

    auto out = 0.0;
    for (auto& e : g.adjacencies[node])
      out += e->current;

    if (!almost_equal(in, out)) {
      std::cout << "Ops\n";
      std::cout << "Node: " << node << "\n";
      std::cout << "In: " << in << ". Out: " << out << "\n";
    }
  }
}

Amperes
calculate_current_kcl(Circuit& g, real Vtotal)
{
  namespace e = Eigen;

  auto m = g.effective_node_count();
  auto& M = g.admittance_matrix;
  auto fst = g.actual_node(0);

  //------ Building coefficient matrix
  // Matrix coefKCL =
  //   -(M.block(fst, fst, m, m) + M.transpose().block(fst, fst, m, m));
  Matrix coefKCL = -(M + M.transpose()).block(fst, fst, m, m);
  coefKCL.diagonal() =
    (M.rowwise().sum() + M.colwise().sum().transpose()).segment(fst, m);

  //------ Building independent term
  auto size = g.levels[2].size();
  Vector currentsKCL = Vector::Zero(m);
  currentsKCL.segment(0, size) =
    (Vtotal * M).colwise().sum().segment(fst, size);

  //------ Removing zeroed rows and columns from coef matrix
  e::Matrix<bool, 1, e::Dynamic> non_zero_cols =
    coefKCL.cast<bool>().colwise().any();

  Matrix A(non_zero_cols.count(), non_zero_cols.count());
  std::vector<std::size_t> keep;
  keep.reserve(coefKCL.rows());

  for (e::Index u = 0, j = 0; u < coefKCL.cols(); u++) {
    if (!non_zero_cols(u))
      continue;

    for (e::Index v = 0, i = 0; v < coefKCL.rows(); v++) {
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
  Vector VV = solver.solve(II);

  Vector V = Vector::Zero(m);
  {
    Eigen::Index i = 0;
    for (auto x : keep) {
      V(x) = VV(i++);
    }
  }

  //------ Calculating currents to branches connected to Vcc
  for (auto node : g.levels[2]) {
    auto i = g.pseudo_node(node);
    for (auto& e : g.inputs[node]) {
      auto I = (Vtotal - V[i]) / e->fuse.R;
      e->current = I;
    }
  }

  //------ Calculating output currents via Ohms law
  for (auto i = 0u; i < m; i++) {
    auto node = g.actual_node(i);
    for (auto& e : g.adjacencies[node]) {
      auto j = g.pseudo_node(e->dst);
      auto Vj = j < m ? V[j] : 0;
      auto I = (V[i] - Vj) / e->fuse.R;
      e->current = I;
    }
  }
  verify_kcl(g);

  //------ Calculating final current
  auto total_current = Amperes(0);

  for (auto node : g.levels[g.levels.size() - 3]) {
    for (auto& e : g.adjacencies[node]) {
      total_current += e->current;
    }
  }

  return total_current;
}

Log
simulation(size_t L, real D, real V0, real deltaV)
{
  auto g = generate_circuit(L, D);
  auto original = g;
  draw_graph(diff_graph(original, g), "begin");
  auto V = V0;
  auto l = Log();
  auto sink = g.nodes.size() - 1;

  std::vector<real> currents(g.nodes.size());

  while (g.connected()) {
    // std::cout << V << "\n";
    // print_la(kcl(g, V));
    auto current = calculate_current_kcl(g, V);
    // calculate_current(g, V);

    iteration(g, V);
    l.log(V, current);
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

  if (argc >= 3) {
    L = std::stoul(argv[1]);
    D = std::stod(argv[2]);
  }
  if (argc >= 4) {
    auto type = std::string{ argv[3] };
    if (type == "s")
      CurrentType = CircuitType::Square;
    else if (type == "t")
      CurrentType = CircuitType::Tilted;
    else if (type == "h")
      CurrentType = CircuitType::Hexagon;
    else
      std::cout << "WARNING: Invalid circuit type\n";
  }

  seed_rand();
  // auto start = now();
  auto s = simulation(L, D, 50, 5);
  // std::cout << s.iterations() << " iterations\n";
  // std::cout << elapsed(start) << " ms\n";
  std::ofstream out{ "output.csv" };
  s.show(out);
}
