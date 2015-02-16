function jitter(img)
    local ang = 6.28
    local angle = torch.uniform(0,ang)
    image.rotate(img,img,angle)

   -- Make all flip cases equallly likely
   -- flip == 1, is no flip
   -- U(0.5,4.5) gives each case equal ranges in dist

    local flip = torch.round(torch.uniform(0.5,4.5))
    
    if flip == 2 then
      image.hflip(img,img)
    elseif flip == 3 then
      image.vflip(img,img)
    elseif flip >= 4 then
      image.hflip(img,img)
      image.vflip(img,img)
    end
   img:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
   return img
end
