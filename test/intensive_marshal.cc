#include <gtest/gtest.h>

#include "shared/internal_marshal.h"
#include "shared/initialize.h"
TEST(normal, intensive_marshal_test) {
  using namespace canary;
  struct evbuffer* buffer = evbuffer_new();
  const int n = 0x100000;
  {
    CanaryOutputArchive output_archive(buffer);
    for (int i = 0; i < n; ++i) {
      output_archive(i * i / 5);
    }
  }
  {
    CanaryInputArchive input_archive(buffer);
    for (int i = 0; i < n; ++i) {
      int j;
      input_archive(j);
      EXPECT_EQ(i * i / 5, j);
    }
  }
  evbuffer_free(buffer);
}

TEST(reference, intensive_marshal_test) {
  using namespace cereal;
  std::stringstream buffer;
  const int n = 0x100000;
  {
    BinaryOutputArchive output_archive(buffer);
    for (int i = 0; i < n; ++i) {
      output_archive(i * i / 5);
    }
  }
  {
    BinaryInputArchive input_archive(buffer);
    for (int i = 0; i < n; ++i) {
      int j;
      input_archive(j);
      EXPECT_EQ(i * i / 5, j);
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::canary::InitializeCanaryWorker(&argc, &argv);
  return RUN_ALL_TESTS();
}
