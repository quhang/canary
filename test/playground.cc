#include <unistd.h>
#include <cstdio>
#include "event2/buffer.h"
int main() {
  struct evbuffer* buffer = evbuffer_new();

  //evbuffer_expand(buffer, 1e7);
  for (int i = 0; i < 1e6; ++i) {
    evbuffer_add_printf(buffer, "hello");
    // printf("(%zu %zu)",
    //        evbuffer_get_length(buffer),
    //        evbuffer_get_contiguous_space(buffer));
  }
  evbuffer_pullup(buffer, 1e7);
  return 0;
}
