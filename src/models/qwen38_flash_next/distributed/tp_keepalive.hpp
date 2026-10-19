#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_KEEPALIVE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_KEEPALIVE_HPP_

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <chrono>

namespace gufo::models::qwen38_flash_next::distributed {

// The TP2 ranks keep two idle TCP sockets: the RDMA bootstrap and the control
// channel. Keepalive only reports a peer host that vanished; a rank that died
// closes its sockets at once. Probing after a minute and giving up after
// twelve unanswered probes lets the pair ride out a network outage of up to
// three minutes, e.g. a rebooting switch, while RDMA keeps working.
inline constexpr int kTpKeepaliveIdleSeconds = 60;
inline constexpr int kTpKeepaliveIntervalSeconds = 10;
inline constexpr int kTpKeepaliveProbes = 12;
inline constexpr auto kTpKeepaliveTolerance = std::chrono::seconds(
    kTpKeepaliveIdleSeconds + kTpKeepaliveIntervalSeconds * kTpKeepaliveProbes);

// How long rank 0 waits for rank 1 to reach the RDMA bootstrap, and rank 1
// keeps retrying to reach rank 0. After a lost pair each rank restarts on its
// own schedule, so both wait long enough to cover the other's restart.
inline constexpr auto kTpBootstrapWindow = std::chrono::seconds(120);

// A frame sent during an outage is resent with a backoff that doubles up to
// two minutes, so after a two-minute outage the next retry can come a minute
// or more after the link returns, and the waiting rank's exchange times out
// first. Capping the backoff (TCP_RTO_MAX_MS, Linux 6.15; older kernels reject
// it) at 20 s resends within 20 s of the link's return. With the default
// tcp_retries2 of 15 the connection then gives up after about 200 s instead
// of about 15 minutes; a smaller cap would give up sooner than that.
inline constexpr int kTcpRtoMaxMsOption = 44;
inline constexpr int kTpRetransmitCapMilliseconds = 20000;

/// Turns on TCP keepalive with the TP2 tolerance above and caps the
/// retransmission backoff where the kernel supports it.
inline void SetTpKeepalive(int fd) noexcept {
  (void)::setsockopt(fd, IPPROTO_TCP, kTcpRtoMaxMsOption,
                     &kTpRetransmitCapMilliseconds,
                     sizeof(kTpRetransmitCapMilliseconds));
  const int on = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on));
#if defined(TCP_KEEPIDLE) && defined(TCP_KEEPINTVL) && defined(TCP_KEEPCNT)
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &kTpKeepaliveIdleSeconds,
                     sizeof(kTpKeepaliveIdleSeconds));
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL,
                     &kTpKeepaliveIntervalSeconds,
                     sizeof(kTpKeepaliveIntervalSeconds));
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &kTpKeepaliveProbes,
                     sizeof(kTpKeepaliveProbes));
#endif
}

}  // namespace gufo::models::qwen38_flash_next::distributed

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_KEEPALIVE_HPP_
