local getRCL = require 'models/rcl'
local p = 0.5

local function getModel()
   local model = nn.Sequential()

   model:add(getRCL(3, 64, 3, 1, 1, 3, 3, true, 'pre'))
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
   model:add(nn.Dropout(p))

   model:add(getRCL(64, 128, 3, 1, 1, 3, 3, true, 'pre'))
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
   model:add(nn.Dropout(p))

   model:add(getRCL(128, 256, 3, 1, 1, 3, 3, true, 'pre'))
   model:add(cudnn.SpatialAveragePooling(8, 8, 8, 8, 0, 0))
   model:add(nn.Dropout(p))

   model:add(nn.Reshape(256))
   model:add(nn.Linear(256, 10))

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.weight:normal(0, 0.01)
      v.bias:zero()
   end

   model:cuda()
   model:get(1).gradInput = nil

   local criterion = nn.CrossEntropyCriterion():cuda()

   return model, criterion
end

return getModel
