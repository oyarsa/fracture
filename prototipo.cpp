#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

using std::size_t;

using time_point = std::chrono::high_resolution_clock::time_point;
using NodeId_t = size_t;
using real = double;

struct Node {
  NodeId_t num;
  size_t level;

  Node(size_t num, size_t level) : num(num), level(level) {}
};

struct Fuse {
  real R;
  real Imax;

  Fuse(real R, real Imax) : R(R), Imax(Imax) {}

  bool blown(real I) { return I > Imax; }
};

struct Edge {
  NodeId_t src;
  NodeId_t dst;
  Fuse fuse;

  Edge(NodeId_t src, NodeId_t dst, Fuse fuse) : src(src), dst(dst), fuse(fuse) {}
};

bool operator==(const Edge &a, const Edge &b) {
  return a.src == b.src && a.dst == b.dst;
}

template <typename C, typename T> auto erase(C &col, const T &val) {
  using std::begin;
  using std::end;
  return col.erase(std::remove(begin(col), end(col), val), end(col));
}

template <typename C, typename I>
auto swap_remove(C& container, I it) {
  using std::swap;
  swap(*it, container.back());
  container.pop_back();
  return it;
}

template <typename C>
auto swap_remove(C& container, size_t index) {
  using std::begin;
  return swap_remove(container, begin(container) + index);
}

struct Circuit {
  std::vector<NodeId_t> nodes;
  std::vector<std::vector<NodeId_t>> levels;
  std::vector<std::vector<Edge>> adjacencies;
  std::vector<std::vector<Edge>> inputs;

  Circuit(size_t n) : nodes(n), levels(), adjacencies(n), inputs(n) {}

  // void add_node(std::unique_ptr<Node> node) {
  void add_node(NodeId_t node, size_t level) {
    add_node_to_level(node, level);
    nodes[node] = node;
  }

  void add_node_to_level(NodeId_t node, size_t level) {
    while (levels.size() <= level) {
      levels.push_back({});
    }
    assert(levels.size() >= level + 1);
    levels[level].push_back(node);
  }

  void add_edge(const Edge &edge) {
    adjacencies[edge.src].push_back(edge);
    inputs[edge.dst].push_back(edge);
  }

  void remove_input_edge(const Edge& edge) {
    // erase(inputs[edge.dst], edge);
    auto& ins = inputs[edge.dst];
    swap_remove(ins, std::find(begin(ins), end(ins), edge));
  }

  bool connected() {
    auto source = nodes.front();
    auto sink = nodes.back();
    std::vector<char> visited(nodes.size(), 0);
    return idfs(source, sink, visited);
  }

  bool idfs(NodeId_t cur, NodeId_t target, std::vector<char> &visited) {
    std::stack<NodeId_t> s;
    s.push(cur);
    while (!s.empty()) {
      cur = s.top();
      s.pop();
      if (cur == target) return true;
      if (!visited[cur]) {
        visited[cur] = 1;
        for (auto& next : adjacencies[cur])
          s.push(next.dst);
      }
    }
    return false;
  }
};

const real inf = 1e9;

std::vector<real> calculate_parallel_current(const Circuit &g, real V) {
  auto R = 0.0;

  for (auto &adj : g.adjacencies) {
    for (auto &e : adj) {
      if (e.fuse.R != 0.0)
        R += 1.0 / e.fuse.R;
    }
  }

  auto I = V * R;
  return std::vector<real>(g.nodes.size(), I);
}

std::vector<real> calculate_polygon_current(const Circuit&g, real V) {
  std::vector<real> currents(g.nodes.size());
  std::vector<real> outputs(g.nodes.size());

  currents[0] = inf;
  outputs[0] = inf;

  for (auto i = 1u; i < g.levels.size(); i++) {
    for (auto& p : g.levels[i]) {
      for (auto& in : g.inputs[p]) {
        currents[p] += outputs[in.src]; // Resistencia impacta como?
      }
      auto num_outs = g.adjacencies[p].size();
      if (num_outs != 0)
        outputs[p] = currents[p] / num_outs; // Corrente distribuída igualmente entre as saídas
    }
  }

  return currents;
}

std::vector<real> calculate_series_current(const Circuit &g, real V) {
  auto R = 0.0;

  for (auto &adj : g.adjacencies) {
    for (auto &e : adj) {
      R += e.fuse.R;
    }
  }

  auto I = V / R;
  return std::vector<real>(g.nodes.size(), I);
}

Circuit generate_parallel_circuit(size_t L, real) {
  auto g = Circuit(L + 2);
  auto init = 0;
  g.add_node(init, 0);

  auto finish = L + 1;

  for (auto i = 1u; i < L; i += 2) {
    auto node_a = i;
    g.add_node(node_a, 1);
    g.add_edge(Edge(init, node_a, Fuse(0, inf)));

    auto node_b = i + 1;
    g.add_node(node_b, 2);
    g.add_edge(Edge(node_a, node_b, Fuse(20, 20 * i)));

    g.add_edge(Edge(node_b, finish, Fuse(0, inf)));
  }

  g.add_node(finish, 3);

  return g;
}

Circuit generate_series_circuit(size_t L, real) {
  auto g = Circuit(L + 2);

  auto init = 0;
  auto prev = init;
  g.add_node(init, 0);

  for (auto i = 1u; i <= L; i++) {
    auto node = i;
    g.add_node(node, i);

    auto R = i == 1 ? 0.0 : 20.0;
    auto Imax = i == 1 ? inf : 20.0;
    g.add_edge( Edge(prev, node, Fuse(R, Imax)));

    prev = node;
  }

  auto finish = L + 1;
  g.add_node(finish, L + 1);
  g.add_edge(Edge(prev, finish, Fuse(0, inf)));

  return g;
}

Circuit generate_square_circuit(size_t L, real D) {
  return Circuit(L);
}

Circuit generate_hexagon_circuit(size_t L, real D) {
  return Circuit(L);
}

enum TipoCircuito { Series, Parallel, Square, Hexagon };
TipoCircuito CurrentType = Parallel;

std::vector<real> calculate_current(const Circuit &g, real V) {
  switch (CurrentType) {
  case Parallel:
    return calculate_parallel_current(g, V);
  case Series:
    return calculate_series_current(g, V);
  case Square:
  case Hexagon:
    return calculate_polygon_current(g, V);
  default:
    return {};
  }
}

Circuit generate_circuit(size_t L, real D) {
  switch (CurrentType) {
  case Parallel:
    return generate_parallel_circuit(L, D);
  case Series:
    return generate_series_circuit(L, D);
  case Square:
    return generate_square_circuit(L, D);
  case Hexagon:
    return generate_hexagon_circuit(L, D);
  default:
    return Circuit(L);
  }
}

struct Log {
  std::vector<std::pair<real, real>> xy;

  void log(real V, real I) { xy.push_back({V, I}); }

  void show(std::ostream &out = std::cout) {
    out << "V,I\n";
    for (auto &p : xy) {
      out << p.first << "," << p.second << "\n";
    }
  }
};

void iteracao(Circuit &g, const std::vector<real>& currents) {
  for (auto &points : g.levels) {
    for (auto &p : points) {
      auto I = currents[p];
      auto &adj = g.adjacencies[p];
      auto it = begin(adj);

      while (it != end(adj)) {
        if (it->fuse.blown(I)) {
          g.remove_input_edge(*it);
          swap_remove(adj, it);
          // std::swap(*it, adj.back());
          // adj.pop_back();
        } else {
          ++it;
        }
      }
    }
  }
}

std::string flt2str(real f) {
  std::ostringstream oss;
  oss << std::defaultfloat << f;
  return oss.str();
}

void draw_graph(const Circuit &g) {
  std::ofstream out{"graph.dot"};
  out << "digraph {\n";
  out << "  graph [rankdir=LR]\n";
  out << "  node [height=0.05 label=\"\" shape=point width=0.05]\n";
  out << "  edge [arrowhead=empty arrowsize=0.5 fontsize=8]\n";

  for (auto &v : g.nodes) {
    for (auto &e : g.adjacencies[v]) {
      auto &f = e.fuse;
      std::string label;
      if (f.R == 0)
        label = "";
      else
        label = flt2str(f.R) + "Ω " + flt2str(f.Imax) + "A";
      out << "    " << v << " -> " << e.dst << "[label=\"" << label
          << "\"]\n";
    }
  }

  out << "}\n";
}

Log simulacao(size_t L, real D, real V0, real deltaV) {
  auto g = generate_circuit(L, D);
  // draw_graph(g);
  auto V = V0;
  auto l = Log();
  auto sink = g.nodes.size() - 1;

  while (g.connected()) {
    auto currents = calculate_current(g, V);
    l.log(V, currents[sink]);

    iteracao(g, currents);
    V += deltaV;
  }

  l.log(V, 0);
  return l;
}

time_point now() { return std::chrono::high_resolution_clock::now(); }

long elapsed(const time_point &t) {
  auto e = now() - t;
  return std::chrono::duration_cast<std::chrono::milliseconds>(e).count();
}

int main() {
  auto start = now();
  auto s = simulacao(400, 0, 0, 0.1);
  std::cout << elapsed(start) << " ms\n";
  std::ofstream out{"saida.csv"};
  s.show(out);
}
