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

template <typename T> auto compare_ptr(const T *p) {
  return [=](const std::unique_ptr<T> &q) { return q.get() == p; };
}

template <typename C, typename T> auto erase(C &col, const T &val) {
  using std::begin;
  using std::end;
  return col.erase(std::remove(begin(col), end(col), val), end(col));
}

template <typename C, typename P> auto erase_if(C &col, P p) {
  using std::begin;
  using std::end;
  return col.erase(std::remove_if(begin(col), end(col), p), end(col));
}

struct Circuit {
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::vector<Node *>> levels;
  std::vector<std::vector<Edge>> adjacencies;

  Circuit(size_t n) : nodes(n), levels(), adjacencies(n) {}

  void add_node(std::unique_ptr<Node> node) {
    add_node_to_level(node.get());
    nodes[node->num] = move(node);
  }

  void add_node_to_level(Node *node) {
    while (levels.size() <= node->level) {
      levels.push_back({});
    }
    assert(levels.size() >= node->level + 1);
    levels[node->level].push_back(node);
  }

  void add_edge(const Edge &edge) {
    adjacencies[edge.src].push_back(edge);
  }

  void remove_node(Node *node) {
    erase(levels[node->level], node);
    adjacencies[node->num].clear();

    for (auto &adj : adjacencies) {
      for (auto it = begin(adj); it != end(adj); ++it) {
        if (it->dst == node->num) {
          adj.erase(it);
          break;
        }
      }
    }

    erase_if(nodes, compare_ptr(node));
  }

  auto remove_edge(const Edge &edge) {
    auto &adj = adjacencies[edge.src];
    return erase(adj, edge);
  }

  bool connected() {
    auto source = nodes.front().get();
    auto sink = nodes.back().get();
    std::vector<char> visited(nodes.size(), 0);
    return idfs(source->num, sink->num, visited);
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

  bool dfs(NodeId_t cur, NodeId_t target, std::vector<char> &visited) {
    if (cur == target)
      return true;

    visited[cur] = true;
    for (auto &next : adjacencies[cur]) {
      if (!visited[next.dst] && dfs(next.dst, target, visited))
        return true;
    }
    return false;
  }
};

const real inf = 1e9;

real calculate_parallel_current(const Circuit &g, real V) {
  auto R = 0.0;

  for (auto &adj : g.adjacencies) {
    for (auto &e : adj) {
      if (e.fuse.R != 0.0)
        R += 1.0 / e.fuse.R;
    }
  }

  return V * R;
}

real calculate_series_current(const Circuit &g, real V) {
  auto R = 0.0;

  for (auto &adj : g.adjacencies) {
    for (auto &e : adj) {
      R += e.fuse.R;
    }
  }

  return V / R;
}

Circuit generate_parallel_circuit(size_t L, real) {
  auto g = Circuit(L + 2);
  auto init = std::make_unique<Node>(0, 0);
  auto init_id = 0;
  g.add_node(move(init));

  auto finish = std::make_unique<Node>(L + 1, 3);
  auto finish_id = L + 1;

  for (auto i = 1u; i < L; i += 2) {
    auto a_id = i;
    auto node_a = std::make_unique<Node>(a_id, 1);
    g.add_node(move(node_a));
    g.add_edge(Edge(init_id, a_id, Fuse(0, inf)));

    auto b_id = i + 1;
    auto node_b = std::make_unique<Node>(i + 1, 2);
    g.add_node(move(node_b));
    g.add_edge(Edge(a_id, b_id, Fuse(20, 20 * i)));

    g.add_edge(Edge(b_id, finish_id, Fuse(0, inf)));
  }

  g.add_node(move(finish));

  return g;
}

Circuit generate_series_circuit(size_t L, real) {
  auto g = Circuit(L + 2);

  auto init = std::make_unique<Node>(0, 0);
  auto prev = 0;
  g.add_node(move(init));

  for (auto i = 1u; i <= L; i++) {
    auto node_id = i;
    auto node = std::make_unique<Node>(node_id, i);
    g.add_node(move(node));

    auto R = i == 1 ? 0.0 : 20.0;
    auto Imax = i == 1 ? inf : 20.0;
    g.add_edge(Edge(prev, node_id, Fuse(R, Imax)));

    prev = node_id;
  }

  auto finish = std::make_unique<Node>(L + 1, L + 1);
  g.add_node(move(finish));
  g.add_edge(Edge(prev, L + 1, Fuse(0, inf)));

  return g;
}

enum TipoCircuito { Series, Parallel };
TipoCircuito CurrentType = Parallel;

real calculate_current(const Circuit &g, real V) {
  switch (CurrentType) {
  case Parallel:
    return calculate_parallel_current(g, V);
  case Series:
    return calculate_series_current(g, V);
  default:
    return 0.0;
  }
}

Circuit generate_circuit(size_t L, real D) {
  switch (CurrentType) {
  case Parallel:
    return generate_parallel_circuit(L, D);
  case Series:
    return generate_series_circuit(L, D);
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

real iteracao(Circuit &g, real V, real deltaV, Log &r) {
  auto I = calculate_current(g, V);

  for (auto &points : g.levels) {
    auto blown = false;

    for (auto &p : points) {
      auto &adj = g.adjacencies[p->num];
      auto it = begin(adj);

      while (it != end(adj)) {
        if (it->fuse.blown(I)) {
          // it = adj.erase(it); // Remove a aresta diretamente.
          std::swap(*it, adj.back());
          adj.pop_back();
          blown = true;
        } else {
          ++it;
        }
      }

      if (blown)
        return V;
    }
  }

  r.log(V, I);
  return V + deltaV;
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
    for (auto &e : g.adjacencies[v->num]) {
      auto &f = e.fuse;
      std::string label;
      if (f.R == 0)
        label = "";
      else
        label = flt2str(f.R) + "Ω " + flt2str(f.Imax) + "A";
      out << "    " << v->num << " -> " << e.dst << "[label=\"" << label
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

  while (g.connected()) {
    V = iteracao(g, V, deltaV, l);
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
  // ofstream out{"saida.csv"};
  // s.show(out);
}
