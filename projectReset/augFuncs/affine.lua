function jitter(s)
 local d = torch.rand(2)
   -- vflip
   if d[1] > 0.5 then
    s = image.vflip(s)
  end
   -- hflip
   if d[2] > 0.5 then
    s = image.hflip(s)
  end
  
  local  T1 = torch.eye(3,3)
  T1[1][3] = -64
  T1[2][3] = -64
  T1[3][3] = 1

  local   S = torch.eye(3,3)
  S[1][2] = torch.rand(1):div(2.5)
  S[2][1] = torch.rand(1):div(2.5)

  local   ang = torch.rand(1):mul(math.pi)
  local   T2 = torch.Tensor(3,3):fill(0)
  T2[1][1] = torch.cos(ang)
  T2[2][1] = torch.sin(ang)
  T2[1][2] = torch.sin(ang):mul(-1)
  T2[2][2] = torch.cos(ang)
  T2[3][3] = 1

  local   T3 = torch.eye(3,3)
  T3[1][3] = torch.rand(1):mul(5):add(-2.5):add(64)
  T3[2][3] = torch.rand(1):mul(5):add(-2.5):add(64)
  T3[3][3] = 1

  local   Sc = torch.eye(3,3):mul(torch.randn(1):div(10):add(0.95)[1])

  local   T = torch.mm(S, T1)
  T = torch.mm(T2, T)
  T = torch.mm(T3, T)
  T = torch.mm(Sc,T)
  T = T[{{1,2}, {1,3}}]

  dest = torch.Tensor(128,128):fill(0)

--  for i = 1,128 do
--   for j=1,128 do
--     local x = math.floor(T[1][1]*i + T[1][2]*j + T[1][3])
--     local y = math.floor(T[2][1]*i + T[2][2]*j + T[2][3])
--     if x > 0 and x <= 128 and y > 0 and y <= 128 then
--       dest[i][j] = s[x][y]
--     end
--   end
-- end
 dest = opencv.WarpAffine(s,T)
 dest = image.scale(dest,sampleSize[2],sampleSize[3])
 dest = dest:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
return dest
end
