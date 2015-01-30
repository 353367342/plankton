require 'torch'
require 'image'

function randomTransform(img,nTrans)
  local imgDim = img:size()
  local imgBatch = torch.Tensor(nTrans,1,imgDim[1],imgDim[2])
--  img = image.lcn(img)
  for i = 1,nTrans do
    local ang = 360
    local angle = torch.uniform(0,ang)
    image.rotate(imgBatch[i],img,angle)

   -- Make all flip cases equallly likely
   -- flip == 1, is no flip
   -- U(0.5,4.5) gives each case equal ranges in dist

    local flip = torch.round(torch.uniform(0.5,4.5))
    
    if flip == 2 then
      image.hflip(imgBatch[i],imgBatch[i])
    elseif flip == 3 then
      image.vflip(imgBatch[i],imgBatch[i])
    elseif flip >= 4 then
      image.hflip(imgBatch[i],imgBatch[i])
      image.vflip(imgBatch[i],imgBatch[i])
    end  
  end
  return imgBatch 
end

--img = randomTransform(image.lena()[1],10) 
