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
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

using std::size_t;

using time_point = std::chrono::high_resolution_clock::time_point;
using NodeId_t = size_t;
using real = double;
using Amperes = real;
using Ohms = real;

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
  std::vector<std::vector<NodeId_t>> levels;
  std::vector<std::vector<std::shared_ptr<Edge>>> adjacencies;
  std::vector<std::vector<std::shared_ptr<Edge>>> inputs;

  Circuit(size_t n)
    : nodes(n)
    , levels()
    , adjacencies(n)
    , inputs(n)
  {
  }

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
    auto e = std::make_shared<Edge>(edge);
    adjacencies[edge.src].push_back(e);
    inputs[edge.dst].push_back(e);
  }

  void remove_input_edge(const std::shared_ptr<Edge>& edge)
  {
    auto& ins = inputs[edge->dst];
    ins.erase(std::remove(begin(ins), end(ins), edge), end(ins));
  }

  void remove_node_inputs(NodeId_t node)
  {
    auto& ins = inputs[node];
    for (auto& in : ins) {
      auto& adj = adjacencies[in->src];
      adj.erase(std::remove(begin(adj), end(adj), in), end(adj));
      if (adj.empty()) {
        remove_node_inputs(in->src);
      }
    }
    ins.clear();
  }

  bool connected()
  {
    auto source = nodes.front();
    return adjacencies[source].size() > 0;
  }

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

Fuse
random_fuse(real D, real R)
{
  if (nextreal() <= D) {
    R *= 0.5 + nextreal();
  }
  return Fuse(R, R);
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
        g.add_edge(Edge(prev[nx], curr[j], random_fuse(D, 20)));
      }
      if (ny < max) {
        g.add_edge(Edge(prev[ny], curr[j], random_fuse(D, 20)));
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

      auto fuseH = random_fuse(D, 20 * horiz);
      fuseH.Imax = 20;
      g.add_edge(Edge(prev[j], curr[j], fuseH));
      if (j > 0) {
        auto fuseV = random_fuse(D, 20 * vert);
        fuseV.Imax = 20;
        g.add_edge(Edge(curr[j - 1], curr[j], fuseV));
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

void
iteration(Circuit& g, real V, std::vector<real>& currents)
{
  // Total current in circuit. Maybe improve this?
  const auto total_current = 20 * V;

  // Distribute the current evenly to the first level.
  for (auto& out : g.adjacencies[0]) {
    out->current = total_current / g.adjacencies[0].size();
  }

  /*
   * For each iteration, per level after the first:
   *   - Calculate input current on point
   *     Sum currents coming from input edges.
   *   - Calculate output currents
   *     Distribute current on point to the output edges, inversely proportional
   *     to the edge's resistance.
   *   - Check if the edge will blow with the current (if it exceeds its Imax)
   *     If it does, remove the edge from the graph.
   *   - Check if the point (except the destination) still has any output edge.
   *     If it doesn't, current must not flow to it, and the input edges must
   *     be removed.
   */
  for (auto lit = begin(g.levels) + 1; lit != end(g.levels); ++lit) {
    for (auto& p : *lit) {
      Amperes current(0.0);
      for (auto& e : g.inputs[p]) {
        current += e->current;
      }

      currents[p] = current;
      auto& adj = g.adjacencies[p];
      auto ratios = calculate_ratios(adj);
      auto it = begin(adj);
      auto i = 0;

      while (it != end(adj)) {
        Amperes I = current * ratios[i++];
        if ((*it)->fuse.blown(I)) {
          g.remove_input_edge(*it);
          // std::cout << current << ", " << adj.size()  << ": " << I << " - " << V << " - "
          //           << (*it)->src << "->" << (*it)->dst << ": " << (*it)->fuse.R << "\n";
          it = adj.erase(it);
        } else {
          (*it)->current = I;
          ++it;
        }
      }

      if (adj.empty() && lit != end(g.levels) - 1) {
        g.remove_node_inputs(p);
      }
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
    out << "    " << e.src << " -> " << e.dst << "[label=\"" << label << e.src
        << "->" << e.dst << "\" "
        << "style=\"" << style << "\" "
        << "arrowhead=\"" << head << "\" "
        << "]\n";
  }

  out << "}\n";

  std::cout << "Graph printed: " << id << "\n";
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
    iteration(g, V, currents);
    l.log(V, currents[sink]);
    V += deltaV;
  }
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
  auto start = now();
  auto s = simulation(L, D, 0, 0.1);
  std::cout << s.iterations() << " iterations\n";
  std::cout << elapsed(start) << " ms\n";
  std::ofstream out{ "output.csv" };
  s.show(out);
}
