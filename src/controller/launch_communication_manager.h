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
 * @file src/controller/launch_communication_manager.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class LaunchCommunicationManager.
 */

#ifndef CANARY_SRC_CONTROLLER_LAUNCH_COMMUNICATION_MANAGER_H_
#define CANARY_SRC_CONTROLLER_LAUNCH_COMMUNICATION_MANAGER_H_

#include <list>
#include <map>
#include <string>

#include "shared/canary_internal.h"

#include "controller/launch_communication_interface.h"
#include "shared/network.h"

namespace canary {

class LaunchCommunicationManager : public LaunchSendCommandInterface {
 private:
  typedef LaunchCommunicationManager SelfType;
  /**
   * The record corresponds to a launcher.
   */
  struct LauncherRecord {
    //! The id of the launching command.
    LaunchCommandId launch_command_id;
    //! Communication socket.
    int socket_fd = -1;
    //! Read event.
    struct event* read_event = nullptr;
    //! Write event.
    struct event* write_event = nullptr;
    //! Send buffer, which is grabbed from the sending queue.
    struct evbuffer* send_buffer = nullptr;
    //! Send queue, which owns all the buffers.
    std::list<struct evbuffer*> send_queue;
    //! Receive buffer.
    struct evbuffer* receive_buffer = nullptr;
    //! Pointer to the manager.
    SelfType* manager = nullptr;
  };

  //! Default backlog argument, i.e. Concurrent pending TCP connecting events.
  static const int kBacklog = -1;

 public:
  //! Constructor.
  LaunchCommunicationManager() {}
  //! Destructor.
  virtual ~LaunchCommunicationManager() {}

  //! Initializes the communication manager.
  // Prepare call.
  void Initialize(network::EventMainThread* event_main_thread,
                  LaunchReceiveCommandInterface* command_receiver,
                  const std::string& launch_service = FLAGS_launch_service);
  //! Shuts down the listener.
  void Finalize();
  //! Responds to a launching command. The buffer ownership is transferred.
  void SendLaunchResponseCommand(LaunchCommandId launch_command_id,
                                 struct evbuffer* buffer) override;

 public:
  /*
   * Public static methods used to dispatch event callbacks.
   */

  //! Dispatches the accept event of the listening controller port.
  // Sync call.
  static void DispatchAcceptEvent(struct evconnlistener* listener,
                                  int socket_fd, struct sockaddr* address,
                                  int socklen, void* arg);
  //! Dispatches the accept error event of the listening controller port.
  // Sync call.
  static void DispatchAcceptErrorEvent(struct evconnlistener*, void*);
  //! Dispatches the read event on a launcher channel.
  // Sync call.
  static void DispatchReadEvent(int socket_fd, short, void* arg);  // NOLINT
  //! Dispatches the write event on a launcher channel.
  // Sync call.
  static void DispatchWriteEvent(int socket_fd, short, void* arg);  // NOLINT

 private:
  /*
   * Core logic to handle accept/read/write events.
   */

  //! Receives the accept event of the listening controller port.
  // Sync call.
  void CallbackAcceptEvent(struct evconnlistener* listener, int socket_fd,
                           struct sockaddr* address, int socklen);
  //! Receives data or error on a socket to a launcher.
  // Sync call.
  void CallbackReadEvent(LaunchCommandId launch_command_id,
                         LauncherRecord* launcher_record);
  //! Sends data to a launcher.
  // Sync call.
  void CallbackWriteEvent(LauncherRecord* launcher_record);
  //! Responds a launching command.
  // Sync call.
  void AppendLaunchResponseCommandSendingQueue(
      LaunchCommandId launch_command_id, struct evbuffer* buffer);

  /*
   * Lifetime of a connection.
   */
  void InitializeLauncherRecord(LaunchCommandId launch_command_id,
                                int socket_fd);
  void CleanUpLauncherRecord(LauncherRecord* launch_command_record);

 private:
  network::EventMainThread* event_main_thread_ = nullptr;
  struct event_base* event_base_ = nullptr;
  LaunchReceiveCommandInterface* command_receiver_ = nullptr;

  bool is_initialized_ = false;

  // Launch command id keeps increasing and is not reused.
  LaunchCommandId launch_command_id_allocator_ = LaunchCommandId::FIRST;
  std::map<LaunchCommandId, LauncherRecord> launch_command_id_to_status_;

  int listening_socket_ = -1;
  struct evconnlistener* listening_event_ = nullptr;
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_LAUNCH_COMMUNICATION_MANAGER_H_
