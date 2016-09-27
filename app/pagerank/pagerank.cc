#include <cereal/archives/xml.hpp>

#include <algorithm>
#include <array>
#include <sstream>
#include <utility>
#include <vector>

#include "canary/canary.h"

static int FLAG_app_partition_x = 1;          // Number of partitions in x.
static int FLAG_app_partition_y = 1;          // Number of partitions in y.
static int FLAG_app_size_per_partition = 3;  // Size per partition.
static int FLAG_app_iterations = 10;          // Number of iterations.

namespace canary {

class PagerankApplication : public CanaryApplication {
 public:
  const double kAlpha = .15;
  typedef uint32_t VertexId;

  /**
   * User defined vertex property.
   */
  struct VertexUserProperty {
    double rank, temp_rank;
    template <class Archive>
    void serialize(Archive& archive) {
      archive(rank, temp_rank);
    }
  };

  /**
   * The property of a vertex.
   */
  template <typename T>
  struct VertexProperty {
    // The pagerank of a vertex.
    T user_property;
    // The outgoing degree of a vertex.
    uint32_t out_degree;
    // Edge partitions that need the vertex value, i.e. routing table in
    // GraphX.
    std::vector<int> routing_receivers;
    // Serialization function.
    template <class Archive>
    void serialize(Archive& archive) {
      archive(user_property, out_degree, routing_receivers);
    }
  };

  /**
   * The property of a vertex, full declaration.
   */
  typedef VertexProperty<VertexUserProperty> VertexFullProperty;

  /**
   * A vertex partition.
   */
  struct VertexPropertyPartition {
    std::unordered_map<VertexId, VertexFullProperty> vertices;
    int num_neighboring_edge_partitions;
    template <class Archive>
    void serialize(Archive& archive) {
      archive(vertices, num_neighboring_edge_partitions);
    }
  };

  /**
   * The compressed row storage of a partition of edges.
   */
  struct EdgePropertyPartition {
    // Row index to vertex id.
    std::vector<VertexId> row_index_to_vertex_id;
    // Row pointer to the first index in the column array.
    std::vector<uint32_t> row_pointer;
    // The column index array.
    std::vector<VertexId> column_index;
    template <class Archive>
    void serialize(Archive& archive) {
      archive(row_index_to_vertex_id, row_pointer, column_index);
    }
  };

  /*
   * The following functions are used to construct a 2-D mesh.
   */
  //! Returns all vertices in a partition.
  std::vector<VertexId> GetAllVertices(int partition_id) const {
    std::vector<VertexId> result;
    const int partition_id_x = partition_id / FLAG_app_partition_y;
    const int partition_id_y = partition_id % FLAG_app_partition_y;
    const int max_y = FLAG_app_partition_y * FLAG_app_size_per_partition;
    for (int cell_x = partition_id_x * FLAG_app_size_per_partition;
         cell_x <= (1 + partition_id_x) * FLAG_app_size_per_partition; ++cell_x)
      for (int cell_y = partition_id_y * FLAG_app_size_per_partition;
           cell_y <= (1 + partition_id_y) * FLAG_app_size_per_partition;
           ++cell_y) {
        result.push_back(cell_x * (max_y + 1) + cell_y);
      }
    return std::move(result);
  }

  //! Returns all outgoing edges of a vertex.
  std::vector<std::pair<VertexId, VertexId>> GetAllOutgoingEdges(
      VertexId vertex_id) const {
    std::vector<std::pair<VertexId, VertexId>> result;
    const int max_x = FLAG_app_partition_x * FLAG_app_size_per_partition;
    const int max_y = FLAG_app_partition_y * FLAG_app_size_per_partition;
    int cell_x = vertex_id / (max_y + 1);
    int cell_y = vertex_id % (max_y + 1);
    if (cell_x > 0) {
      result.emplace_back(vertex_id, vertex_id - (max_y + 1));
    }
    if (cell_x < max_x) {
      result.emplace_back(vertex_id, vertex_id + (max_y + 1));
    }
    if (cell_y > 0) {
      result.emplace_back(vertex_id, vertex_id - 1);
    }
    if (cell_y < max_y) {
      result.emplace_back(vertex_id, vertex_id + 1);
    }
    return std::move(result);
  }

  //! Returns the partition id of an edge.
  int GetPartitionIdOfEdge(VertexId from, VertexId to) const {
    if (from > to) {
      std::swap(from, to);
    }
    const int max_y = FLAG_app_partition_y * FLAG_app_size_per_partition;
    const int cell_x = from / (max_y + 1);
    const int cell_y = from % (max_y + 1);
    const int partition_id_x = std::min(cell_x / FLAG_app_size_per_partition,
                                        FLAG_app_partition_x - 1);
    const int partition_id_y = std::min(cell_y / FLAG_app_size_per_partition,
                                        FLAG_app_partition_y - 1);
    return partition_id_y + partition_id_x * FLAG_app_partition_y;
  }

  int GetNumNeighbor(int partition_id) const {
    const int partition_id_x = partition_id / FLAG_app_partition_y;
    const int partition_id_y = partition_id % FLAG_app_partition_y;
    int len_x = 1, len_y = 1;
    if (partition_id_x > 0) {
      ++len_x;
    }
    if (partition_id_x < FLAG_app_partition_x - 1) {
      ++len_x;
    }
    if (partition_id_y > 0) {
      ++len_y;
    }
    if (partition_id_y < FLAG_app_partition_y - 1) {
      ++len_y;
    }
    if (partition_id_x < FLAG_app_partition_x - 1 &&
       partition_id_y < FLAG_app_partition_y - 1) {
      // The bottom right partition does not need to communicate.
      return len_x * len_y - 2;
    } else {
      return len_x * len_y - 1;
    }
  }

  // The program.
  void Program() override {
    const int FLAG_app_partitions = FLAG_app_partition_x * FLAG_app_partition_y;
    auto d_vertex_partition =
        DeclareVariable<VertexPropertyPartition>(FLAG_app_partitions);
    auto d_edge_partition = DeclareVariable<EdgePropertyPartition>();
    auto d_sum_rank = DeclareVariable<double>(1);

    WriteAccess(d_vertex_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto vertex_partition = task_context->WriteVariable(d_vertex_partition);
      std::set<int> neighboring_edge_partitions;
      for (auto vertex_id : GetAllVertices(task_context->GetPartitionId())) {
        auto& vertex_property = vertex_partition->vertices[vertex_id];
        vertex_property.user_property.temp_rank = 0.;
        vertex_property.user_property.rank = 1.;
        const auto outgoing_edges = GetAllOutgoingEdges(vertex_id);
        vertex_property.out_degree = outgoing_edges.size();
        std::set<int> buffer;
        for (auto edge : outgoing_edges) {
          buffer.insert(GetPartitionIdOfEdge(edge.first, edge.second));
        }
        buffer.erase(task_context->GetPartitionId());
        // If another partition owns the edge, then a routing entry is set up.
        vertex_property.routing_receivers.insert(
            vertex_property.routing_receivers.end(),
            buffer.begin(), buffer.end());
      }
      vertex_partition->num_neighboring_edge_partitions =
        GetNumNeighbor(task_context->GetPartitionId());
    });

    WriteAccess(d_edge_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto edge_partition = task_context->WriteVariable(d_edge_partition);
      for (auto vertex_id : GetAllVertices(task_context->GetPartitionId())) {
        edge_partition->row_index_to_vertex_id.push_back(vertex_id);
        edge_partition->row_pointer.push_back(
            edge_partition->column_index.size());
        const auto outgoing_edges = GetAllOutgoingEdges(vertex_id);
        for (auto edge : outgoing_edges) {
          int edge_partition_id = GetPartitionIdOfEdge(edge.first, edge.second);
          if (edge_partition_id == task_context->GetPartitionId()) {
            edge_partition->column_index.push_back(edge.second);
          }
        }
      }
      edge_partition->row_pointer.push_back(
          edge_partition->column_index.size());
    });

    Loop(FLAG_app_iterations);

    TrackNeeded();
    ReadAccess(d_edge_partition);
    WriteAccess(d_vertex_partition);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& edge_partition = task_context->ReadVariable(d_edge_partition);
      auto vertex_partition = task_context->WriteVariable(d_vertex_partition);
      const auto& row_index_to_vertex_id =
          edge_partition.row_index_to_vertex_id;
      const auto& row_pointer = edge_partition.row_pointer;
      const auto& column_index = edge_partition.column_index;
      // Iterate over all edges.
      for (size_t index = 0; index < row_index_to_vertex_id.size(); ++index) {
        const VertexId left_vertex = row_index_to_vertex_id[index];
        const double contribute =
            vertex_partition->vertices[left_vertex].user_property.rank /
            vertex_partition->vertices[left_vertex].out_degree;
        for (size_t to_index = row_pointer[index];
             to_index < row_pointer[index + 1]; ++to_index) {
          const VertexId right_vertex = column_index[to_index];
          vertex_partition->vertices[right_vertex].user_property.temp_rank +=
              contribute;
        }
      }
    });

    typedef std::vector<std::pair<VertexId, double>> ScatterMessage;

    ReadAccess(d_vertex_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& vertex_partition =
          task_context->ReadVariable(d_vertex_partition);
      std::map<int, ScatterMessage> buffers;
      for (const auto& key_value : vertex_partition.vertices) {
        const auto& vertex_property = key_value.second;
        for (auto destination : vertex_property.routing_receivers) {
          buffers[destination].emplace_back(
              key_value.first, vertex_property.user_property.temp_rank);
        }
      }
      for (auto& key_value : buffers) {
        task_context->Scatter(key_value.first, key_value.second);
      }
    });

    WriteAccess(d_vertex_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto vertex_partition = task_context->WriteVariable(d_vertex_partition);
      EXPECT_GATHER_SIZE(vertex_partition->num_neighboring_edge_partitions);
      auto buffers = task_context->Gather<ScatterMessage>();
      for (auto& buffer : buffers) {
        for (auto& key_value : buffer) {
          vertex_partition->vertices[key_value.first].user_property.temp_rank +=
              key_value.second;
        }
      }
      return 0;
    });

    WriteAccess(d_vertex_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto vertex_partition = task_context->WriteVariable(d_vertex_partition);
      for (auto& key_value : vertex_partition->vertices) {
        auto& vertex_property = key_value.second;
        vertex_property.user_property.rank =
            vertex_property.user_property.temp_rank * kAlpha +
            vertex_property.user_property.rank * (1 - kAlpha);
        vertex_property.user_property.temp_rank = 0.;
      }
    });

    EndLoop();

    // Computes the sum of ranks.
    ReadAccess(d_vertex_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& vertex_partition =
          task_context->ReadVariable(d_vertex_partition);
      double sum = 0;
      for (auto& key_value : vertex_partition.vertices) {
        auto& vertex_property = key_value.second;
        int edge_partition_id = GetPartitionIdOfEdge(key_value.first,
                                                     key_value.first);
        if (edge_partition_id == task_context->GetPartitionId()) {
          sum += vertex_property.user_property.rank;
        }
      }
      task_context->Scatter(0, sum);
    });

    WriteAccess(d_sum_rank);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      const double sum = task_context->Reduce(0., std::plus<double>());
      printf("%f \n", sum);
      fflush(stdout);
      return 0;
    });
  }

  void LoadParameter(const std::string& parameter) override {
    std::stringstream ss;
    ss << parameter;
    {
      cereal::XMLInputArchive archive(ss);
      LoadFlag("partition_x", FLAG_app_partition_x, archive);
      LoadFlag("partition_y", FLAG_app_partition_y, archive);
      LoadFlag("size_per_partition", FLAG_app_size_per_partition, archive);
      LoadFlag("iterations", FLAG_app_iterations, archive);
    }
  }
};

}  // namespace canary

REGISTER_APPLICATION(::canary::PagerankApplication);
