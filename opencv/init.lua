require 'torch'
require 'image'

opencv = {}

require 'libopencv'

function opencv.WarpAffine(source, warp)
  local img = source
  local dest = torch.Tensor():resizeAs(img)
  img.libopencv.WarpAffine(img,dest,warp)
  return dest
end