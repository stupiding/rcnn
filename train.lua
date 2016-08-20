local optim = require 'optim'


local function top1Error(output, labels)
   local _, predictions = output:sort(2, true)
   predictions = predictions[{{}, {1}}]
   local nError = predictions:ne(labels:long()):sum()
   return nError
end


local TrainNet = torch.class('torch.TrainNet')

function TrainNet:__init(opt, model, criterion)
   self.chunkSize = opt.chunkSize
   self.testSize = opt.testSize
   self.nGPU = opt.nGPU

   self.model = model
   self.criterion = criterion

   self.optimState = {
      learningRate = opt.lr,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
end

function TrainNet:trainClassify(lr, dataProvider, chunkSize)
   local timer = torch.Timer()
   params, gradParams = self.model:getParameters()

   if lr then
      self.optimState.learningRate = lr
   end
   local chunkSize = chunkSize or self.chunkSize or dataProvider.epochSize

   self.model:training()
   local function opFunc()
         return self.criterion.output, gradParams
   end
   local N = torch.ceil(chunkSize / dataProvider.batchSize)
   local actualChunkSize = N * dataProvider.batchSize
   --local mod = chunkSize - (N - 1) * dataProvider.batchSize
   local lossSum, nError = 0, 0

   for n = 1, N do
      local batch = dataProvider:getBatch()
      self:loadBatch(batch)

      local output = self.model:forward(self.input):float()
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()

      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(opFunc, params, self.optimState)

      lossSum = lossSum + loss
      nError = nError + top1Error(output, batch.labels)

      --assert(self.params:storage() == self.model:parameters()[1].storage())
   end
   local avgLoss = lossSum / N
   local errorRate = nError / actualChunkSize
   local t = timer:time().real
   return avgLoss, errorRate, t
end

function TrainNet:testClassify(dataProvider, chunkSize)
   local timer = torch.Timer()

   local chunkSize = chunkSize or self.testSize or dataProvider.epochSize

   self.model:evaluate()
   local N = torch.ceil(chunkSize / dataProvider.batchSize)
   local mod = chunkSize - (N - 1) * dataProvider.batchSize
   local nError = 0

   for n = 1, N do
      local batch = dataProvider:getBatch()
      self:loadBatch(batch)

      local output = self.model:forward(self.input):float()

      if n < N then
         nError = nError + top1Error(output, batch.labels)
      else
         nError = nError + top1Error(
            output[{{1, mod}, {}}],
            batch.labels[{{1, mod}}]
         )
      end
   end
   self.model:training()

   local errorRate = nError / chunkSize
   local t = timer:time().real
   return errorRate, t
end

function TrainNet:loadBatch(batch)
   self.input = self.input or (self.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(batch.samples:size()):copy(batch.samples)
   self.target:resize(batch.labels:size()):copy(batch.labels)
end

return torch.TrainNet
