#include <TH.h>
#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define libopencv_(NAME) TH_CONCAT_3(libopencv_, Real, NAME)

#include "warpAffine.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libopencv(lua_State *L) {
  libopencv_FloatMain_init(L);
  libopencv_DoubleMain_init(L);
  return 1;
}
