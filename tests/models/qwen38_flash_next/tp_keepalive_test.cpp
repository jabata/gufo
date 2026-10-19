#include "src/models/qwen38_flash_next/distributed/tp_keepalive.hpp"

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

namespace distributed = gufo::models::qwen38_flash_next::distributed;

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

int IntOption(int fd, int level, int name) {
  int value = -1;
  socklen_t size = sizeof(value);
  Require(::getsockopt(fd, level, name, &value, &size) == 0, "getsockopt");
  return value;
}

}  // namespace

int main() {
  // A rebooting home gateway took the LAN down for 92 s and 119 s; the pair
  // must outlast that, and each rank's restart must find the other.
  Require(distributed::kTpKeepaliveTolerance >= std::chrono::minutes(3),
          "keepalive gives up before a three-minute outage ends");
  Require(distributed::kTpBootstrapWindow >= std::chrono::minutes(2),
          "bootstrap window is shorter than a peer restart");

  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  Require(fd >= 0, "socket");
  distributed::SetTpKeepalive(fd);
  Require(IntOption(fd, SOL_SOCKET, SO_KEEPALIVE) != 0, "keepalive is off");
#if defined(TCP_KEEPIDLE) && defined(TCP_KEEPINTVL) && defined(TCP_KEEPCNT)
  Require(IntOption(fd, IPPROTO_TCP, TCP_KEEPIDLE) ==
              distributed::kTpKeepaliveIdleSeconds,
          "keepalive idle time");
  Require(IntOption(fd, IPPROTO_TCP, TCP_KEEPINTVL) ==
              distributed::kTpKeepaliveIntervalSeconds,
          "keepalive interval");
  Require(IntOption(fd, IPPROTO_TCP, TCP_KEEPCNT) ==
              distributed::kTpKeepaliveProbes,
          "keepalive probe count");
#endif
  int cap = -1;
  socklen_t cap_size = sizeof(cap);
  if (::getsockopt(fd, IPPROTO_TCP, distributed::kTcpRtoMaxMsOption, &cap,
                   &cap_size) == 0) {
    Require(cap == distributed::kTpRetransmitCapMilliseconds,
            "retransmission backoff is not capped");
  } else {
    std::puts("kernel has no TCP_RTO_MAX_MS; backoff cap not checked");
  }
  ::close(fd);
  std::puts("PASS");
  return 0;
}
