#include <array>
#include <algorithm>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "simulation-mpi.h"

#include "shared/internal.h"
#include "application/canary-context.h"
#include "util/data-object-helper.h"
#include "util/message-helper.h"

DEFINE_int32(app_cell_x, 10, "Number of cells in the x dimension.");
DEFINE_int32(app_cell_y, 10, "Number of cells in the y dimension.");
DEFINE_int32(app_cell_z, 10, "Number of cells in the z dimension.");
DEFINE_int32(app_partition_x, 1, "Number of partitions in the x dimension.");
DEFINE_int32(app_partition_y, 1, "Number of partitions in the y dimension.");
DEFINE_int32(app_partition_z, 1, "Number of partitions in the z dimension.");
DEFINE_int32(app_frames, 10, "Number of maximum frames.");

// Data object operations.
using ::canary::TypedDataObject;
using ::canary::DataObjectHelper;
// Message operations.
using ::canary::MessageHelper;
// Task function input data types.
using ::canary::WriteVector;
using ::canary::ReadVector;
using ::canary::TaskContext;
using ::canary::SendMessageContainer;
using ::canary::ReceiveMessageContainer;
// Partition index type.
using ::canary::PartitionIndex;

using ::Lassen::SimulationMPI;

struct GlobalState {
  std::array<double, 4> extreme_buffer;
  bool is_global_done = false;
  double dt = 0;
  int iteration = 0;
  int frame = 0;
  void save(::canary::OutputArchive& archive) const {  // NOLINT
    archive(extreme_buffer, is_global_done, dt, iteration, frame);
  }
  void load(::canary::InputArchive& archive) {  // NOLINT
    archive(extreme_buffer, is_global_done, dt, iteration, frame);
  }
};

int main(int argc, char *argv[]) {
  ::canary::CanaryInitializeInternal(argc, argv);
  ::canary::CanaryContext cc;

  typedef TypedDataObject<SimulationMPI> PartitionDataObject;
  typedef TypedDataObject<GlobalState> GlobalDataObject;

  auto d_partition =
      cc.DeclareMDD(PartitionDataObject::GetPrototype("partition"),
                    "partition");
  auto d_global =
      cc.DeclareMDD(GlobalDataObject::GetPrototype("global"), "global");

  cc.Calc([] F_CALCULATE {
    GlobalState& global = DataObjectHelper::get<GlobalState>(write_vector[0]);
    global.frame = FLAGS_app_frames;
  }, "InitializeGlobalObject").Modify(d_global);

  cc.Calc([] F_CALCULATE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    sim.metadata.initialize(
        task_context.self_num_partition, task_context.self_partition_index,
        FLAGS_app_partition_x, FLAGS_app_partition_y, FLAGS_app_partition_z,
        FLAGS_app_cell_x, FLAGS_app_cell_y, FLAGS_app_cell_z);
    CHECK(sim.metadata.checkConsistency());
    sim.Setup();
    sim.functionLocalInitialize();
  }, "InitializeSimulationObject").Modify(d_partition);

  typedef std::array<double, 4> FourDoubleArray;
  cc.Send([] F_SEND {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    FourDoubleArray buffer;
    sim.functionSendSyncExtreme(buffer.data());
    send_container->AddMessage<FourDoubleArray>(buffer);
  }, "ReduceSendSyncExtreme").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    GlobalState& global = DataObjectHelper::get<GlobalState>(write_vector[0]);
    CHECK_GE(task_context.pair_num_partition,
             (int)recv_container->message_list.size());
    size_t required_message =
        task_context.pair_num_partition - recv_container->message_list.size();
    if (required_message == 0) {
      auto& reduce_result = global.extreme_buffer;
      std::fill(reduce_result.begin(), reduce_result.end(),
                std::numeric_limits<double>::max());
      for (auto& message_ptr : recv_container->message_list) {
        auto element =
            MessageHelper<FourDoubleArray>::ExtractMessage(message_ptr);
        for (size_t i = 0; i < element.size(); ++i) {
          reduce_result[i] = std::min(reduce_result[i], element[i]);
        }
      }
    }
    return required_message;
  }, "ReduceRecvSyncExtreme").Modify(d_global);

  cc.Send([] F_SEND {
    GlobalState& global = DataObjectHelper::get<GlobalState>(write_vector[0]);
    for (PartitionIndex index = 0; index < task_context.pair_num_partition;
         ++index) {
      send_container->AddMessage<FourDoubleArray>(global.extreme_buffer, index);
    }
  }, "BroadcastSendSyncExtreme").Modify(d_global);

  cc.Recv([] F_RECEIVE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    CHECK_GE(1, (int)recv_container->message_list.size());
    size_t required_message = 1 - recv_container->message_list.size();
    for (auto& message_ptr : recv_container->message_list) {
      auto buffer = MessageHelper<FourDoubleArray>::ExtractMessage(message_ptr);
      sim.functionRecvSyncExtreme(buffer.data());
      // std::cout << canary::to_debug_string(buffer) << std::endl;
    }
    return required_message;
  }, "BroadcastRecvSyncExtreme").Modify(d_partition);

  cc.Calc([] F_CALCULATE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    sim.functionInitializeBoundaryNodes();
    sim.functionInitializeCommunicatingNodes();
  }, "InitializeNodes").Modify(d_partition);

  cc.Send([] F_SEND {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    auto& index_array = sim.domain->neighborDomains;
    std::vector<long> sendBuffer;
    sim.functionSendInitializeNodeCommunication(sendBuffer);
    for (int index : index_array) {
      send_container->AddMessage<decltype(sendBuffer)>(sendBuffer, index);
    }
  }, "SendInitializeNodeCommunication").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    auto& index_array = sim.domain->neighborDomains;
    CHECK_GE(index_array.size(), recv_container->message_list.size());
    size_t required_message
        = index_array.size() - recv_container->message_list.size();
    if (required_message == 0) {
      std::vector<std::vector<long>> recvBuffer(index_array.size());
      for (auto& message_ptr : recv_container->message_list) {
        for (size_t i = 0; i < index_array.size(); ++i) {
          if (message_ptr.get_message_header()->from_partition_index
              == index_array[i]) {
            recvBuffer[i]
                = MessageHelper<std::vector<long>>::ExtractMessage(message_ptr);
            break;
          }
        }
      }
      // long sum = 0;
      // for (auto& i : recvBuffer) for (auto& j : i) sum += j;
      // std::cout << sum << std::endl;
      sim.functionRecvInitializeNodeCommunication(recvBuffer);
    }
    return required_message;
  }, "RecvInitializeNodeCommunication").Modify(d_partition);

  cc.Send([] F_SEND {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    Lassen::BoundingBox boundingBox;
    sim.functionSendFirstInitializeFacetCommunication(boundingBox);
    for (PartitionIndex index = 0; index < task_context.pair_num_partition;
         ++index) {
      send_container->AddMessage<decltype(boundingBox)>(boundingBox, index);
    }
  }, "SendInitializeBoxCommunication").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    CHECK_GE(task_context.pair_num_partition,
             (int)recv_container->message_list.size());
    size_t required_message
        = task_context.pair_num_partition - recv_container->message_list.size();
    if (required_message == 0) {
      CHECK_EQ(sim.domain->numDomains, task_context.pair_num_partition);
      sim.allBoundingBox.resize(sim.domain->numDomains);
      for (auto& message_ptr : recv_container->message_list) {
        const auto receiving_index =
            message_ptr.get_message_header()->from_partition_index;
        sim.allBoundingBox[receiving_index]
            = MessageHelper<Lassen::BoundingBox>::ExtractMessage(message_ptr);
      }
      // for (auto& i : sim.allBoundingBox) {
      //   printf("%f %f %f %f %f %f;\n",
      //          i.min.x, i.min.y, i.min.z, i.max.x, i.max.y, i.max.z);
      // }
      sim.functionRecvFirstInitializeFacetCommunication(sim.allBoundingBox);
    }
    return required_message;
  }, "RecvInitializeBoxCommunication").Modify(d_partition);

  cc.Send([] F_SEND {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    auto& index_array = sim.overlapDomains;
    std::vector<Lassen::Point> sendBuffer(sim.boundaryNodes.size());
    sim.functionSendSecondInitializeFacetCommunication(sendBuffer);
    for (int index : index_array) {
      send_container->AddMessage<decltype(sendBuffer)>(sendBuffer, index);
    }
  }, "SendInitializeFacetCommunication").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    auto& index_array = sim.overlapDomains;
    CHECK_GE(index_array.size(), recv_container->message_list.size());
    size_t required_message =
        index_array.size() - recv_container->message_list.size();
    if (required_message == 0) {
      std::vector<std::vector<Lassen::Point>> recvBuffer(index_array.size());
      for (auto& message_ptr : recv_container->message_list) {
        for (size_t i = 0; i < index_array.size(); ++i) {
          if (message_ptr.get_message_header()->from_partition_index
              == index_array[i]) {
            recvBuffer[i]
                = MessageHelper<std::vector<Lassen::Point>>::ExtractMessage(
                    message_ptr);
            break;
          }
        }
      }
      // double sum = 0;
      // for (auto& i : recvBuffer) for (auto& j : i) sum += j.x + j.y + j.z;
      // printf("%f\n", sum);
      sim.functionRecvSecondInitializeFacetCommunication(recvBuffer);
    }
    return required_message;
  }, "RecvInitializeFacetCommunication").Modify(d_partition);

  cc.Send([] F_SEND {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    auto& index_array = sim.overlapDomains;
    sim.functionSendThirdInitializeFacetCommunication();
    for (size_t index = 0; index < index_array.size(); ++index) {
      send_container->AddMessage<int>(sim.isFacetNeighbor[index],
                                      index_array[index]);
    }
  }, "SendInitializeThird").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    auto& index_array = sim.overlapDomains;
    CHECK_GE(index_array.size(), recv_container->message_list.size());
    size_t required_message =
        index_array.size() - recv_container->message_list.size();
    if (required_message == 0) {
      std::vector<int> recvIsFacetNeighbor(sim.overlapDomains.size(), 0);
      for (auto& message_ptr : recv_container->message_list) {
        for (size_t i = 0; i < index_array.size(); ++i) {
          if (message_ptr.get_message_header()->from_partition_index
              == index_array[i]) {
            recvIsFacetNeighbor[i] =
                MessageHelper<int>::ExtractMessage(message_ptr);
            break;
          }
        }
      }
      // long sum = 0;
      // for (auto& i : recvIsFacetNeighbor) {
      //   sum *= 2;
      //   sum += i;
      // }
      // printf("%ld\n", sum);
      sim.functionRecvThirdInitializeFacetCommunication(recvIsFacetNeighbor);
    }
    return required_message;
  }, "RecvInitializeThird").Modify(d_partition);

  cc.Send([] F_SEND {
    send_container->AddMessage<bool>(true);
  }, "BarrierReduceSend").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    size_t required_message =
        task_context.pair_num_partition - recv_container->message_list.size();
    return required_message;
  }, "BarrierReduceRecv").Modify(d_global);

  cc.Send([] F_SEND {
    for (PartitionIndex index = 0; index < task_context.pair_num_partition;
         ++index) {
      send_container->AddMessage<bool>(true, index);
    }
  }, "BarrierBroadcastSend").Modify(d_global);

  cc.Recv([] F_RECEIVE {
    size_t required_message = 1 - recv_container->message_list.size();
    return required_message;
  }, "BroadcastBroadcastRecv").Modify(d_partition);

  cc.While([] F_CONDITION {
    GlobalState& global = DataObjectHelper::get<GlobalState>(write_vector[0]);
    ++global.iteration;
    if (global.iteration % 10 == 0) {
      printf("iteration=%d\n", global.iteration);
    }
    return !global.is_global_done && (global.iteration != global.frame + 1);
  }, "HaveMoreIterations").Modify(d_global).SetBreakpoint();

  cc.Calc([] F_CALCULATE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    sim.functionPrepareStep();
    sim.functionUpdateSources();
    sim.functionUpdateNarrowband();
    sim.functionUpdateNearbyNodes();
    sim.functionConstructFront();
  }, "CalcStepOne").Modify(d_partition);

  cc.Send([] F_SEND {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    auto& index_array = sim.facetNeighbors;
    std::vector<std::vector<double> > sendBuffer;
    sim.functionSendCommunicateFront(sendBuffer);
    for (size_t index = 0; index < index_array.size(); ++index) {
      send_container->AddMessage<std::vector<double>>(
          sendBuffer[index], index_array[index]);
    }
  }, "SendStepOne").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    auto& index_array = sim.facetNeighbors;
    CHECK_GE(index_array.size(), recv_container->message_list.size());
    size_t required_message =
        index_array.size() - recv_container->message_list.size();
    if (required_message == 0) {
      std::vector<std::vector<double>> recvBuffer(index_array.size());
      for (auto& message_ptr : recv_container->message_list) {
        // bool flag = false;
        for (size_t i = 0; i < index_array.size(); ++i) {
          if (message_ptr.get_message_header()->from_partition_index
              == index_array[i]) {
            recvBuffer[i] =
                MessageHelper<std::vector<double>>::ExtractMessage(message_ptr);
            // flag = true;
            break;
          }
        }
        // CHECK(flag);
      }
      // double sum = 0;
      // for (auto& i : recvBuffer) for (auto& j : i) sum += j;
      // printf("%f,", sum);
      sim.functionRecvCommunicateFront(recvBuffer);
    }
    return required_message;
  }, "RecvStepOne").Modify(d_partition);

  cc.Calc([] F_CALCULATE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    // std::stringstream ss;
    // {
    // ::canary::OutputArchive oarchive(ss);
    // oarchive(sim);
    // }
    // sim.~SimulationMPI();
    // new(&sim) SimulationMPI();
    // {
    // ::canary::InputArchive iarchive(ss);
    // iarchive(sim);
    // }
    sim.functionConstructDistancesToFront();
    sim.functionConvertNearbyNodesToNarrowBandNodes();
    sim.functionComputeNextTimeStep();
  }, "CalcStepTwo").Modify(d_partition);

  cc.Send([] F_SEND {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    send_container->AddMessage<double>(sim.dt);
  }, "ReduceSendStepTwo").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    GlobalState& global = DataObjectHelper::get<GlobalState>(write_vector[0]);
    CHECK_GE(task_context.pair_num_partition,
             (int)recv_container->message_list.size());
    size_t required_message =
        task_context.pair_num_partition - recv_container->message_list.size();
    if (required_message == 0) {
      auto& reduce_result = global.dt;
      reduce_result = std::numeric_limits<double>::max();
      for (auto& message_ptr : recv_container->message_list) {
        auto element = MessageHelper<double>::ExtractMessage(message_ptr);
        reduce_result = std::min(reduce_result, element);
      }
    }
    return required_message;
  }, "ReduceRecvStepTwo").Modify(d_global);

  cc.Send([] F_SEND {
    GlobalState& global = DataObjectHelper::get<GlobalState>(write_vector[0]);
    for (PartitionIndex index = 0; index < task_context.pair_num_partition;
         ++index) {
      send_container->AddMessage<double>(global.dt, index);
    }
  }, "BroadcastSendStepTwo").Modify(d_global);

  cc.Recv([] F_RECEIVE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    CHECK_GE(1, (int)recv_container->message_list.size());
    size_t required_message = 1 - recv_container->message_list.size();
    for (auto& message_ptr : recv_container->message_list) {
      auto buffer = MessageHelper<double>::ExtractMessage(message_ptr);
      sim.dt = buffer;
    }
    // printf("%f,", sim.dt);
    return required_message;
  }, "BroadcastRecvStepTwo").Modify(d_partition);

  cc.Calc([] F_CALCULATE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    sim.functionComputeTimeIntegral();
  }, "CalcStepThree").Modify(d_partition);

  cc.Send([] F_SEND {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    auto& index_array = sim.domain->neighborDomains;
    std::vector<std::vector<Lassen::NodeData> > sendBuffer;
    sim.functionSendSynchronizeNodeData(sendBuffer);
    for (size_t index = 0; index < index_array.size(); ++index) {
      send_container->AddMessage<std::vector<Lassen::NodeData>>(
          sendBuffer[index], index_array[index]);
    }
  }, "SendStepThree").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    auto& index_array = sim.domain->neighborDomains;
    CHECK_GE(index_array.size(), recv_container->message_list.size());
    size_t required_message =
        index_array.size() - recv_container->message_list.size();
    if (required_message == 0) {
      std::vector<std::vector<Lassen::NodeData>> recvBuffer(index_array.size());
      for (auto& message_ptr : recv_container->message_list) {
        for (size_t i = 0; i < index_array.size(); ++i) {
          if (message_ptr.get_message_header()->from_partition_index
              == index_array[i]) {
            recvBuffer[i] =
                MessageHelper<std::vector<Lassen::NodeData>>::
                    ExtractMessage(message_ptr);
            break;
          }
        }
        sim.functionRecvSynchronizeNodeData(recvBuffer);
      }
    }
    return required_message;
  }, "RecvStepThree").Modify(d_partition);

  cc.Send([] F_SEND {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    int localDone = 0;
    sim.functionSendDoneCondition(localDone);
    send_container->AddMessage<int>(localDone);
  }, "ReduceSendStepFour").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    GlobalState& global = DataObjectHelper::get<GlobalState>(write_vector[0]);
    CHECK_GE(task_context.pair_num_partition,
             (int)recv_container->message_list.size());
    size_t required_message =
        task_context.pair_num_partition - recv_container->message_list.size();
    if (required_message == 0) {
      int globalDone = 0;
      for (auto& message_ptr : recv_container->message_list) {
        globalDone += MessageHelper<int>::ExtractMessage(message_ptr);
      }
      global.is_global_done = (globalDone == task_context.pair_num_partition);
    }
    return required_message;
  }, "ReduceRecvStepFour").Modify(d_global);

  cc.EndWhile();

  cc.Send([] F_SEND {
    SimulationMPI& sim = DataObjectHelper::get<SimulationMPI>(write_vector[0]);
    std::array<double, 4> buffer;
    sim.functionSendComputeError(
        &buffer[0], &buffer[1], &buffer[2], &buffer[3]);
    send_container->AddMessage<decltype(buffer)>(buffer);
  }, "ReduceSendFinal").Modify(d_partition);

  cc.Recv([] F_RECEIVE {
    CHECK_GE(task_context.pair_num_partition,
             (int)recv_container->message_list.size());
    size_t required_message =
        task_context.pair_num_partition - recv_container->message_list.size();
    if (required_message == 0) {
      std::array<double, 4> result;
      std::fill(result.begin(), result.end(), 0);
      for (auto& message_ptr : recv_container->message_list) {
        auto element = MessageHelper<std::array<double, 4>>::ExtractMessage(
            message_ptr);
        result[0] += element[0];
        result[1] += element[1];
        result[2] = std::max(result[2], element[2]);
        result[3] += element[3];
      }
      result[0] /= result[3];
      result[1] /= result[3];
      std::cout << "==========================================================="
                   "=====================\n";
      std::cout << "nodeCount  = " << std::round(result[3]) << "\n";
      std::cout << "L1   error = " << result[0] << "\n";
      std::cout << "L2   error = " << result[1] << "\n";
      std::cout << "LInf error = " << result[2] << "\n";
      std::cout << "==========================================================="
                   "=====================\n";
    }
    return required_message;
  }, "ReduceRecvFinal").Modify(d_global);

  cc.Launch({{d_partition,
            FLAGS_app_partition_x *
            FLAGS_app_partition_y *
            FLAGS_app_partition_z}}, std::string());
  return 0;
}
