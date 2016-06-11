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
 * @file src/shared/internal_network.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Networking facilities.
 */

#ifndef CANARY_SRC_SHARED_INTERNAL_NETWORK_H_
#define CANARY_SRC_SHARED_INTERNAL_NETWORK_H_

#include <string>

#include "shared/internal_header.h"

// Forward declaration.
struct addrinfo;
struct sockaddr;

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

}  // namespace network
}  // namespace canary
#endif  // CANARY_SRC_SHARED_INTERNAL_NETWORK_H_
