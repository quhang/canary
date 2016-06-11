/*
 * Copyright 2015 Stanford University.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither the name of the copyright holders nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @file src/shared/internal_network.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Networking facilities.
 */

#include "shared/internal_network.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "shared/internal_header.h"

namespace canary {

namespace network {

int close_socket(int socket_fd) { return close(socket_fd); }

int get_last_error_number() { return errno; }

int get_socket_error_number(int socket_fd) {
  int value = 0;
  socklen_t size = sizeof(value);
  PCHECK(getsockopt(socket_fd, SOL_SOCKET, SO_ERROR, &value, &size) == 0);
  // Double checks the return value is integer.
  CHECK_EQ(size, sizeof(value));
  return value;
}

bool is_blocked(int error_number) {
  return error_number == EAGAIN || error_number == EWOULDBLOCK;
}

bool is_blocked() { return is_blocked(get_last_error_number()); }

char* get_error_message(int error_number) { return strerror(error_number); }

int make_socket_nonblocking(int socket_fd) {
  const int flags = fcntl(socket_fd, F_GETFL, nullptr);
  // The flag is always non-negative.
  CHECK_GE(flags, 0);
  if (!(flags & O_NONBLOCK)) {
    if (fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK) == -1) {
      return -1;
    }
  }
  return 0;
}

int make_socket_closeonexec(int socket_fd) {
  const int flags = fcntl(socket_fd, F_GETFD, nullptr);
  // The flag is always non-negative.
  CHECK_GE(flags, 0);
  if (!(flags & FD_CLOEXEC)) {
    if (fcntl(socket_fd, F_SETFD, flags | FD_CLOEXEC) == -1) {
      return -1;
    }
  }
  return 0;
}

int make_socket_nodelay(int socket_fd) {
  const int one = 1;
  return setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &one,
                    static_cast<socklen_t>(sizeof(one)));
}

int make_listen_socket_reuseable(int socket_fd) {
  const int one = 1;
  return setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &one,
                    static_cast<socklen_t>(sizeof(one)));
}

void translate_sockaddr_to_string(struct sockaddr* address, int socklen,
                                  std::string* host, std::string* service) {
  char hostname[NI_MAXHOST];
  char servicename[NI_MAXSERV];
  const int status =
      getnameinfo(address, socklen, hostname, NI_MAXHOST, servicename,
                  NI_MAXSERV, NI_NUMERICHOST | NI_NUMERICSERV);
  CHECK_EQ(status, 0) << gai_strerror(status);
  if (host) {
    host->assign(hostname);
  }
  if (service) {
    service->assign(servicename);
  }
}

int get_socket_local_address_name(int socket_fd, std::string* host,
                                  std::string* service) {
  struct sockaddr_storage address_buffer;
  socklen_t size_buffer = sizeof(address_buffer);
  if (getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address_buffer),
                  &size_buffer) == -1) {
    return -1;
  }
  translate_sockaddr_to_string(reinterpret_cast<sockaddr*>(&address_buffer),
                               size_buffer, host, service);
  return 0;
}

int get_socket_peer_address_name(int socket_fd, std::string* host,
                                 std::string* service) {
  struct sockaddr_storage address_buffer;
  socklen_t size_buffer = sizeof(address_buffer);
  if (getpeername(socket_fd, reinterpret_cast<sockaddr*>(&address_buffer),
                  &size_buffer) == -1) {
    return -1;
  }
  translate_sockaddr_to_string(reinterpret_cast<sockaddr*>(&address_buffer),
                               size_buffer, host, service);
  return 0;
}

void initialize_addrinfo(struct addrinfo* hints, bool is_passive) {
  memset(hints, 0, sizeof(*hints));
  hints->ai_family = AF_INET;
  hints->ai_socktype = SOCK_STREAM;
  hints->ai_flags = is_passive ? AI_PASSIVE : 0;
  hints->ai_protocol = 0;
  hints->ai_canonname = nullptr;
  hints->ai_addr = nullptr;
  hints->ai_next = nullptr;
}

int allocate_and_bind_listen_socket(const std::string& service) {
  int result_fd = -1;
  struct addrinfo hints;
  initialize_addrinfo(&hints, true);
  struct addrinfo* available_addresses = nullptr;
  const int errorcode =
      getaddrinfo(nullptr, service.c_str(), &hints, &available_addresses);
  CHECK_EQ(errorcode, 0) << gai_strerror(errorcode);
  CHECK_NOTNULL(available_addresses);
  result_fd =
      socket(available_addresses->ai_family, available_addresses->ai_socktype,
             available_addresses->ai_protocol);
  PCHECK(result_fd >= 0);
  PCHECK(make_socket_nonblocking(result_fd) == 0);
  PCHECK(make_socket_closeonexec(result_fd) == 0);
  PCHECK(make_socket_nodelay(result_fd) == 0);
  PCHECK(make_listen_socket_reuseable(result_fd) == 0);
  PCHECK(bind(result_fd, available_addresses->ai_addr,
              available_addresses->ai_addrlen) == 0);
  freeaddrinfo(available_addresses);
  return result_fd;
}

int allocate_and_connect_socket(const std::string& host,
                                const std::string& service) {
  int result_fd = -1;
  struct addrinfo hints;
  initialize_addrinfo(&hints, false);
  struct addrinfo* available_addresses = nullptr;
  const int errorcode =
      getaddrinfo(host.c_str(), service.c_str(), &hints, &available_addresses);
  CHECK_EQ(errorcode, 0) << gai_strerror(errorcode);
  CHECK_NOTNULL(available_addresses);
  result_fd =
      socket(available_addresses->ai_family, available_addresses->ai_socktype,
             available_addresses->ai_protocol);
  PCHECK(result_fd >= 0);
  PCHECK(make_socket_nonblocking(result_fd) == 0);
  PCHECK(make_socket_closeonexec(result_fd) == 0);
  PCHECK(make_socket_nodelay(result_fd) == 0);
  const int status = connect(result_fd, available_addresses->ai_addr,
                             available_addresses->ai_addrlen);
  freeaddrinfo(available_addresses);
  if (status != -1 || get_last_error_number() == EINPROGRESS) {
    return result_fd;
  } else {
    close_socket(result_fd);
    return -1;
  }
}

}  // namespace network
}  // namespace canary
