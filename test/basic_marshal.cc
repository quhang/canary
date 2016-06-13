#include <gtest/gtest.h>

#include <vector>
#include <string>

#include "shared/internal_marshal.h"
#include "shared/initialize.h"

TEST(multi_types, basic_marshal_test) {
  using namespace canary;
  struct evbuffer* buffer = evbuffer_new();
  const int a1 = 1;
  const std::string a2 = "hello";
  const std::vector<int> a3 = {4, 5, 6};
  {
    CanaryOutputArchive output_archive(buffer);
    output_archive(a1, a2, a3);
  }
  int b1;
  std::string b2;
  std::vector<int> b3;
  {
    CanaryInputArchive input_archive(buffer);
    input_archive(b1, b2, b3);
  }
  EXPECT_EQ(a1, b1);
  EXPECT_EQ(a2, b2);
  EXPECT_EQ(a3, b3);
  evbuffer_free(buffer);
}

TEST(id, basic_marshal_test) {
  using namespace canary;
  struct evbuffer* buffer = evbuffer_new();
  const ApplicationId a1 = ApplicationId::FIRST;
  const ApplicationId a2 = get_next(a1);
  EXPECT_NE(a1, a2);
  const ApplicationId a3 = get_next(a2);
  EXPECT_NE(a2, a3);
  {
    CanaryOutputArchive output_archive(buffer);
    output_archive(a1, a2, a3);
  }
  ApplicationId b1, b2, b3;
  {
    CanaryInputArchive input_archive(buffer);
    input_archive(b1, b2, b3);
  }
  EXPECT_EQ(a1, b1);
  EXPECT_EQ(a2, b2);
  EXPECT_EQ(a3, b3);
  evbuffer_free(buffer);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::canary::InitializeCanaryWorker(&argc, &argv);
  return RUN_ALL_TESTS();
}
