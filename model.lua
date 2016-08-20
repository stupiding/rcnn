local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local paths = require 'paths'

local function getModel(opt)
   local model, criterion
   local filePath = opt.resume or opt.loadModel
   if filePath then
      assert(paths.filep(filePath), 'Pretrained model file does not exist')
      local file = torch.load(filePath)
      model, criterion = file.model, file.criterion
   else
      local modelPath = paths.concat('models', opt.model .. '.lua')
      assert(paths.filep(modelPath), 'Model file does not exist')
      local modelFunc = require('models/' .. opt.model)
      model, criterion = modelFunc(opt)
   end
   
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   cudnn.fastest = true
   cudnn.benchmark = true
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = true, true
         end)
      dpt.gradInput = nil
      model = dpt:cuda()
   end

   criterion = criterion:cuda()

   return model, criterion
end

return getModel
