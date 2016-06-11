#include "shared/internal.h"
#include "shared/initialize.h"

DEFINE_string(server_service, "19991", "The server service name.");

// 16M buffer.
const int kBufferSize = 0x1000000;
char buffer[kBufferSize];

void BounceBackMessage(int socket_fd) {
  using namespace canary;
  ssize_t n_bytes = 0;
  do {
    do {
      n_bytes = read(socket_fd, buffer, kBufferSize);
    } while (n_bytes == -1 && network::is_blocked());
    if (n_bytes > 0) {
      ssize_t m_bytes = 0;
      size_t to_send = n_bytes;
      char* send_buffer = buffer;
      do {
        do {
          m_bytes = write(socket_fd, send_buffer, to_send);
        } while (m_bytes == -1 && network::is_blocked());
        if (m_bytes == -1) {
          // Writing pipe is broken.
          break;
        }
        send_buffer += m_bytes;
        to_send -= m_bytes;
      } while (to_send != 0);
    }
  } while (n_bytes > 0);
  PCHECK(network::close_socket(socket_fd) == 0);
}

int main(int argc, char* argv[]) {
  using namespace canary;
  InitializeCanaryWorker(&argc, &argv);
  int socket_fd = network::allocate_and_bind_listen_socket(
      FLAGS_server_service);
  PCHECK(listen(socket_fd, 100) == 0);
  while (true) {
    int recv_socket_fd;
    do {
      recv_socket_fd = accept(socket_fd, nullptr, nullptr);
    } while (recv_socket_fd == -1 && network::is_blocked());
    PCHECK(recv_socket_fd != -1);

    std::string buffer_host, buffer_service;

    PCHECK(network::get_socket_local_address_name(
            recv_socket_fd, &buffer_host, &buffer_service) >= 0);
    LOG(INFO) << "Local address: " << buffer_host << " : " << buffer_service;

    PCHECK(network::get_socket_peer_address_name(
            recv_socket_fd, &buffer_host, &buffer_service) >= 0);
    LOG(INFO) << "Peer address: " << buffer_host << " : " << buffer_service;

    BounceBackMessage(recv_socket_fd);
  }
  return 0;
}
