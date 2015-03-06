function jitter(s)
   local d = torch.rand(10)
   -- vflip
   if d[1] > 0.5 then
      s = image.vflip(s)
   end
   -- hflip
   if d[2] > 0.5 then
      s = image.hflip(s)
   end
   -- rotation
   if d[3] > 0.5 then
      s = image.rotate(s, math.pi * d[4],'bilinear')
--      s[torch.eq(s,0)] = 1
   end
   -- crop a 0.9 to 1.1 sized random patch and resize it to 128
   if d[5] > 0.5 then
      local scalef = torch.uniform(0.95, 1.02) 
      local size = {1, sampleSize[2] * scalef, sampleSize[3] * scalef}
      local startX = math.ceil(d[6] * 5)
      local startY = math.ceil(d[7] * 5)
      local endX = startX + size[2]
      local endY = startY + size[3]
      s = image.crop(s, startX, startY, endX, endY)
      -- now rescale it to sampleSize
      s = image.scale(s, sampleSize[2], sampleSize[3],'bilinear')
   else
      -- crop a sampleSize[2]xsampleSize[3] random patch
      local startX = math.ceil(d[6] * 5)
      local startY = math.ceil(d[7] * 5)
      local endX = startX + sampleSize[2]
      local endY = startY + sampleSize[3]
      s = image.crop(s, startX, startY, endX, endY)
   end
   s = s:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
   return s

end
