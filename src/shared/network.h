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
 * @file src/shared/network.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Networking facility.
 */

#ifndef CANARY_SRC_SHARED_NETWORK_H_
#define CANARY_SRC_SHARED_NETWORK_H_

// Libevent header files.
#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/listener.h>

// Networking related header files.
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>

#include "shared/canary_internal.h"

//! Adds one function to libevent: deep copy a buffer.
inline int evbuffer_deep_copy(struct evbuffer* dst, struct evbuffer* src) {
  auto length = evbuffer_get_length(src);
  return evbuffer_add(dst, evbuffer_pullup(src, length), length);
}

namespace canary {
namespace network {

/**
 * Closes a socket. Returns 0 if success, and otherwise -1 with errno set.
 */
int close_socket(int socket_fd);

/**
 * Gets the last error number. Errno is thread-safe, i.e. its implementation is
 * likely a function macro.
 */
int get_last_error_number();

/**
 * Gets the last error number for a socket.
 */
int get_socket_error_number(int socket_fd);

/**
 * Checks whether an error number means blocking.
 */
bool is_blocked(int error_number);

/**
 * Checks whether the last error number means blocking.
 */
bool is_blocked();

/**
 * Gets the error message corresponding to a error number.
 */
char* get_error_message(int error_number);

/**
 * Makes a socket non-blocking. Returns 0 if success, and otherwise -1 with
 * errno set.
 */
int make_socket_nonblocking(int socket_fd);

/**
 * Sets the close-on-exec flag of a socket. Returns 0 if success, and otherwise
 * -1 with errno set.
 */
int make_socket_closeonexec(int socket_fd);

/**
 * Turns off the Nagle algorithm of a socket, and returns 0 is success, and
 * otherwise -1 with errno set.
 */
int make_socket_nodelay(int socket_fd);

/**
 * Makes the address of a listener socket immediately reuseable after the socket
 * is closed. Returns 0 if success, and otherwise -1 with errno set.
 */
int make_listen_socket_reuseable(int socket_fd);

/**
 * Translates a socket address to human-readable host and service names. Fails
 * if the translation does not work.
 */
void translate_sockaddr_to_string(struct sockaddr* address, int socklen,
                                  std::string* host, std::string* service);

/**
 * Gets the local address name of a socket. Returns 0 if success, and -1
 * otherwise with errno set.
 */
int get_socket_local_address_name(int socket_fd, std::string* host,
                                  std::string* service);

/**
 * Gets the peer address name of a socket. Returns 0 if success, and -1
 * otherwise with errno set.
 */
int get_socket_peer_address_name(int socket_fd, std::string* host,
                                 std::string* service);

/**
 * Initializes addrinfo.
 */
void initialize_addrinfo(struct addrinfo* hints, bool is_passive);

/**
 * Allocates and binds a listening socket. Returns the socket fd or fails.
 */
int allocate_and_bind_listen_socket(const std::string& service);

/**
 * Allocates a socket and connects to the other end. Returns the socket fd, and
 * otherwise -1 with errno set.
 */
int allocate_and_connect_socket(const std::string& host,
                                const std::string& service);

/**
 * Sends as much data as possible from the buffer and the queue, and returns
 * the left buffer if sending does not complete.
 */
template <typename Container>
inline struct evbuffer* send_as_much(int socket_fd,
                                     struct evbuffer* send_buffer,
                                     Container* send_queue) {
  while (send_buffer != nullptr || !send_queue->empty()) {
    if (send_buffer == nullptr) {
      send_buffer = send_queue->front();
      send_queue->pop_front();
    }
    const int status = evbuffer_write(send_buffer, socket_fd);
    if (status == -1) {
      break;
    }
    // No need to drain again, evbuffer_write already drains.
    // CHECK_EQ(evbuffer_drain(send_buffer, status), 0);
    // A message is sent.
    if (evbuffer_get_length(send_buffer) == 0) {
      evbuffer_free(send_buffer);
      send_buffer = nullptr;
    }
  }
  return send_buffer;
}

/**
 * Event main thread, a wrapper over event_base.
 */
class EventMainThread {
 protected:
  typedef std::function<void()> CallbackType;

 public:
  EventMainThread();
  virtual ~EventMainThread();

  //! Gets the underlying event base.
  struct event_base* get_event_base() {
    return event_base_;
  }

  //! Returns -1 if error happened, 0 if exit successfully, 1 if exit without
  // any pending events.
  int Run();

  //! Adds an injected handle to be run.
  template <typename T>
  void AddInjectedEvent(T&& handle) {
    const auto status = event_base_once(
        event_base_, 0, EV_TIMEOUT, &DispatchInjectedEvent,
        new CallbackType(std::forward<T>(handle)), zero_timeval_);
    CHECK_EQ(status, 0);
  }

  //! Dispatches an injected event.
  static void DispatchInjectedEvent(int, short, void* arg);  // NOLINT

  //! Adds a delayed injected handle to be run.
  template <typename T>
  void AddDelayInjectedEvent(T&& handle) {
    const auto status = event_base_once(
        event_base_, 0, EV_TIMEOUT, &DispatchInjectedEvent,
        new CallbackType(std::forward<T>(handle)), delay_timeval_);
    CHECK_EQ(status, 0);
  }

  //! Dispatches a delayed injected event.
  static void DispatchDelayInjectedEvent(int, short, void* arg);  // NOLINT

  //! Adds a timeout event after a certain delay.
  template <typename T>
  void AddTimeoutEvent(T&& handle, int delay_sec) {
    struct timeval delay {
      delay_sec, 0
    };
    const auto status =
        event_base_once(event_base_, 0, EV_TIMEOUT, &DispatchInjectedEvent,
                        new CallbackType(std::forward<T>(handle)), &delay);
    CHECK_EQ(status, 0);
  }

 private:
  struct event_base* event_base_ = nullptr;
  // Zero time interval, which is used to indicate a single queue in
  // event_base_.
  const struct timeval* zero_timeval_ = nullptr;
  // One second time interval, which is used to indicate a single queue in
  // event_base_.
  const struct timeval* delay_timeval_ = nullptr;
};

}  // namespace network
}  // namespace canary

#endif  // CANARY_SRC_SHARED_NETWORK_H_
