local cutorch = require 'cutorch'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')
local paths = require 'paths'

local parse = require 'parse'
local getModel = require 'model'
local getDataProvider = require 'data'
local TrainNet = require 'train'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = parse(arg)
local startChunk = 1
local scores = {}
if opt.resume then
   assert(paths.filep(opt.resume, 'Resume file does not exist'))
   local resume = torch.load(opt.resume)
   opt = resume.opt
   startChunk = resume.chunk and resume.chunk + 1 or startChunk
   scores = resume.scores
end

torch.manualSeed(opt.seed)
for i = 1, opt.nGPU do
   cutorch.manualSeed(opt.seed + i, i)
end
cutorch.setDevice(1)

local model, criterion = getModel(opt)
local trainDP, testDP = getDataProvider(opt)
local trainNet = TrainNet(opt, model, criterion)

if opt.testOnly then
   local top1Err, t = trainNet:testClassify(testDP, testDP.epochSize)
   print(string.format('Top1 Error: %.2f. Elapsed time: .2f seconds.', top1Err * 100, t))
   return
end

local function lrFunc(chunk)
   if chunk <= opt.nChunks * 0.5 then
      return opt.lr
   elseif chunk <= opt.nChunks * 0.75 then
      return opt.lr * 0.1
   else
      return opt.lr * 0.01
   end
end

print('Epoch (lr)\t\tTrain Loss\tTrain Error\tTest Error\tElapsed Time')
model:clearState()
for chunk = startChunk, opt.nChunks do
   local lr = lrFunc(chunk)
   local trainLoss, trainTop1, trainTime = trainNet:trainClassify(lr, trainDP)
   local testTop1, testTime = trainNet:testClassify(testDP)
   table.insert(scores, {trainLoss, trainTop1, testTop1})

   if not paths.dirp(opt.saveDir) then
      paths.mkdir(opt.saveDir)
   end

   model:clearState()
   local saveModel = model
   if torch.type(saveModel) == 'nn.DataParallelTable' then
      saveModel = saveModel:get(1)
   end
   torch.save(paths.concat(opt.saveDir, string.format('%d.t7', chunk)),
      {opt = opt, model = saveModel, criterion = criterion, scores = scores, chunk = chunk})

   print(string.format('Chunk %d (%.6f):\t%.6f\t%.2f%%\t\t%.2f%%\t\t%.0fs',
      chunk, lr, trainLoss, trainTop1 * 100, testTop1 * 100, trainTime + testTime))
end
