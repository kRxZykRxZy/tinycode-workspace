// Intel integrated graphics backend of the native runtime.
//
// The OpenVINO path is optional; it is only compiled when OpenVINO is
// available.  The Python layer verifies the provider before it claims
// GPU acceleration.
#include "tinycode/backend.hpp"

#if defined(TINYCODE_WITH_OPENVINO)

#include <openvino/openvino.hpp>

namespace tinycode {
namespace openvino {

std::string provider_name() {
  try {
    ov::Core core;
    const auto devices = core.get_available_devices();
    return devices.empty() ? std::string("none") : devices.front();
  } catch (const std::exception&) {
    return std::string("none");
  }
}

}  // namespace openvino
}  // namespace tinycode

extern "C" const char* tinycode_openvino_provider(void) {
  static const std::string name = tinycode::openvino::provider_name();
  return name.c_str();
}

#endif  // TINYCODE_WITH_OPENVINO
