#include <cuda_runtime_api.h>

namespace deta {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace deta
