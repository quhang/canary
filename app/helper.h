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
 * @file app/helper.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class Helper.
 */

#ifndef CANARY_APP_HELPER_H_
#define CANARY_APP_HELPER_H_

#include <array>

namespace helper {

/*
 * For the best performance, all the functions are in place.
 */

// output += input.
template <typename T, size_t size>
inline void array_add(const std::array<T, size>& input,
                      std::array<T, size>* output) {
  auto input_iter = input.cbegin();
  auto output_iter = output->begin();
  while (input_iter != input.cend()) {
    *output_iter += *input_iter;
    ++input_iter;
    ++output_iter;
  }
}

// output += input.
template <typename T, size_t size>
inline void array_sub(const std::array<T, size>& input,
                      std::array<T, size>* output) {
  auto input_iter = input.cbegin();
  auto output_iter = output->begin();
  while (input_iter != input.cend()) {
    *output_iter -= *input_iter;
    ++input_iter;
    ++output_iter;
  }
}

// output += factor * input.
template <typename T, size_t size>
inline void array_acc(const std::array<T, size>& input, T input_factor,
                      std::array<T, size>* output) {
  auto input_iter = input.cbegin();
  auto output_iter = output->begin();
  while (input_iter != input.cend()) {
    *output_iter += input_factor * (*input_iter);
    ++input_iter;
    ++output_iter;
  }
}

// output = factor * input.
template <typename T, size_t size>
inline void array_mul(const std::array<T, size>& input, T input_factor,
                      std::array<T, size>* output) {
  auto input_iter = input.cbegin();
  auto output_iter = output->begin();
  while (input_iter != input.cend()) {
    *output_iter = input_factor * (*input_iter);
    ++input_iter;
    ++output_iter;
  }
}

// Dot multiply.
template <typename T, size_t size>
inline T array_dot(const std::array<T, size>& left,
                   const std::array<T, size>& right) {
  T result = 0;
  auto left_iter = left.cbegin();
  auto right_iter = right.cbegin();
  while (left_iter != left.cend()) {
    result += (*left_iter) * (*right_iter);
    ++left_iter;
    ++right_iter;
  }
  return result;
}

// Square.
template <typename T, size_t size>
inline T array_square(const std::array<T, size>& input) {
  T result = 0;
  auto iter = input.cbegin();
  while (iter != input.cend()) {
    result += (*iter) * (*iter);
    ++iter;
  }
  return result;
}

// result = ||left - right||
template <typename T, size_t size>
T array_distance(const std::array<T, size>& left,
                 const std::array<T, size>& right) {
  T result = 0;
  auto left_iter = left.cbegin();
  auto right_iter = right.cbegin();
  while (left_iter != left.cend()) {
    const T delta = (*left_iter) - (*right_iter);
    result += delta * delta;
    ++left_iter;
    ++right_iter;
  }
  return result;
}

}  // namespace helper
#endif  // CANARY_APP_HELPER_H_
