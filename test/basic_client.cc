#include <event2/event.h>

#include "shared/canary_internal.h"

#include "shared/initialize.h"
#include "shared/network.h"

DEFINE_string(server_host, "localhost", "The server host name.");
DEFINE_string(server_service, "19991", "The server service name.");

void SendEchoLoop(int socket_fd, const char* buffer, size_t size) {
  using namespace canary;
  {
    auto send_buffer = buffer;
    auto to_send = size;
    ssize_t n_bytes;
    do {
      do {
        n_bytes = write(socket_fd, send_buffer, to_send);
      } while (n_bytes == -1 && network::is_blocked());
      PCHECK(n_bytes != -1);
      send_buffer += n_bytes;
      to_send -= n_bytes;
    } while (to_send != 0);
  };
  char* comp_buffer = new char[size];
  {
    auto recv_buffer = comp_buffer;
    auto to_recv = size;
    do {
      ssize_t n_bytes;
      do {
        n_bytes = read(socket_fd, recv_buffer, to_recv);
      } while (n_bytes == -1 && network::is_blocked());
      PCHECK(n_bytes != -1);
      recv_buffer += n_bytes;
      to_recv -= n_bytes;
    } while (to_recv != 0);
  };
  CHECK_STREQ(buffer, comp_buffer);
  delete[] comp_buffer;
}

void cb_func(int socket_fd, short, void*) {
  using namespace canary;
  auto error_number = network::get_socket_error_number(socket_fd);
  if (error_number == 0) {
    LOG(INFO) << "Pass.";
  } else {
    LOG(FATAL) << network::get_error_message(error_number);
  }
}

int main(int argc, char* argv[]) {
  using namespace canary;
  InitializeCanaryWorker(&argc, &argv);
  int socket_fd = network::allocate_and_connect_socket(
      FLAGS_server_host, FLAGS_server_service);
  PCHECK(socket_fd != -1);

  auto base = event_base_new();
  auto event = event_new(base, socket_fd, EV_WRITE, cb_func, nullptr);
  event_add(event, nullptr);
  event_base_loop(base, EVLOOP_ONCE);
  event_free(event);
  event_base_free(base);

  std::string buffer_host, buffer_service;

  PCHECK(network::get_socket_local_address_name(
          socket_fd, &buffer_host, &buffer_service) == 0);
  LOG(INFO) << "Local address: " << buffer_host << " : " << buffer_service;

  PCHECK(network::get_socket_peer_address_name(
          socket_fd, &buffer_host, &buffer_service) >= 0);
  LOG(INFO) << "Peer address: " << buffer_host << " : " << buffer_service;

  const char message[] = "hello";
  SendEchoLoop(socket_fd, message, sizeof(message));
  PCHECK(network::close_socket(socket_fd) == 0);
  return 0;
}
