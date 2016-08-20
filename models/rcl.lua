local Sequential = nn.Sequential
local Convolution = cudnn.SpatialConvolution
local BN = nn.SpatialBatchNormalization
local ReLU = cudnn.ReLU
local Avg = cudnn.SpatialAveragePooling
local Max = nn.SpatialMaxPooling
local Identity = nn.Identity
local Concat = nn.Concat
local ConcatTable = nn.ConcatTable
local Add = nn.CAddTable
local Dropout = nn.Dropout

local function getRCL(nIn, nOut, nIter, fDepth, rDepth, fSiz, rSiz, share, bn, rDrop, fDrop, drop)
   nIter = nIter or 3
   fDepth = fDepth or 1
   rDepth = rDepth or 1
   fSiz = fSiz or 3
   rSiz = rSiz or 3
   share = (share == true) or false
   bn = bn or 'pre'
   if bn ~= 'pre' then
      bn = 'post'
   end

   local function concatAdd(net, m1, m2)
      net:add(ConcatTable():add(m1):add(m2)):add(Add())
   end

   local function getBlock(nIn, nOut, depth, siz, bn)
      local pad = (siz - 1) / 2

      if depth == 0 then
         if nIn == nOut then
            return Identity()
         else
            return Sequential():add(Concat(2):add(Identity()):add(MulConstant(0)))
         end
      elseif depth == 1 then
         if bn == 'post' then
            return Convolution(nIn, nOut, siz, siz, 1, 1, pad, pad)
         else
            local block = Sequential()
            block:add(Convolution(nIn, nOut, siz, siz, 1, 1, pad, pad))
               :add(BN(nOut))
            return block
         end
      else
         local block = Sequential()
         --[[block:add(Convolution(nIn, nOut, siz, siz, 1, 1, pad, pad))
         if bn == 'pre' then
            block:add(BN(nOut))
         end
         for i = 2, depth do
            if bn == 'post' then
               block:add(BN(nOut))
            end
            block:add(ReLU(true))
            block:add(Convolution(nOut, nOut, siz, siz, 1, 1, pad, pad))
            if bn == 'pre' then
               block:add(BN(nOut))
            end
         end]]

         local longPath = Sequential()
         longPath:add(Convolution(nIn, nOut, siz, siz, 1, 1, pad, pad))
         if bn == 'pre' then
            longPath:add(BN(nOut))
         end
         for i = 2, depth do
            if bn == 'post' then
               longPath:add(BN(nOut))
            end
            longPath:add(ReLU(true))
            longPath:add(Convolution(nOut, nOut, siz, siz, 1, 1, pad, pad))
            if bn == 'pre' then
               longPath:add(BN(nOut))
            end
         end
         local shortPath = Identity()
         concatAdd(block, longPath, shortPath)

         return block
      end
   end

   local nets = {}
   local rBlock1
   for i = 1, nIter do
      local net = Sequential()
      local rec = Sequential()
      local rBlock = getBlock(nOut, nOut, rDepth, rSiz, bn)
      if share then
         if i == 1 then
            rec:add(BN(nOut))
            rec:add(ReLU(true))

            rBlock1 = rBlock
         else
            rec:add(nets[i - 1])

            if torch.typename(rBlock) == 'cudnn.SpatialConvolution' then
               rBlock:share(rBlock1, 'weight', 'bias', 'gradWeight', 'gradBias')
            else
               for j = 1, #rBlock do
                  if torch.typename(rBlock:get(j)) == 'cudnn.SpatialConvolution' then
                     rBlock:get(j):share(rBlock1:get(j), 'weight', 'bias', 'gradWeight', 'gradBias')
                  end
               end
            end
         end
         rec:add(rBlock)
         if rDrop then
            rec:add(Dropout(rDrop))
         end
         if bn == 'post' then
            concatAdd(net, rec, Identity())
         else
            concatAdd(net, rec, BN(nOut))
         end
      else
         local fBlock = getBlock(nIn, nOut, fDepth, fSiz, bn)
         if i == 1 then
            local fInit = getBlock(nIn, nOut, fDepth, fSiz, bn)
            rec:add(fInit)
            if bn == 'post' then
               rec:add(BN(nOut))
            end
            rec:add(ReLU(true))
         else
            rec:add(nets[i - 1])
         end
         rec:add(rBlock)
         concatAdd(net, rec, fBlock)
      end

      if bn == 'post' then
         net:add(BN(nOut))
      end
      net:add(ReLU(true))

      table.insert(nets, net)
   end

   if share then
      local fInit = getBlock(nIn, nOut, fDepth, fSiz, 'post')
      return Sequential():add(fInit):add(nets[#nets])
   else
      return  nets[#nets]
   end
end

return getRCL
