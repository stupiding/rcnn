local datasets = require 'datasets'
local paths = require 'paths'

local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')


local DataProvider = torch.class('torch.DataProvider')

function DataProvider:__init(opt, dataset, split)
   self.pool = threads.Threads(
      opt.nThreads,
      function()
         require 'datasets'
      end,
      function(threadid)
         torch.manualSeed(opt.seed + threadid)
         threadDataset = dataset
      end
   )

   self.batchSize = opt.batchSize

   self.epochSize = dataset:size()
   self.count = 0
   if split == 'train' then
      self.split = split
      self.shuffle = opt.shuffle
      self.idcs = torch.randperm(self.epochSize)
   else
      self.split = 'test'
      self.idcs = torch.range(1, self.epochSize)
   end
end

local batch = nil
function DataProvider:getBatch()
   while self.pool:acceptsjob() do
      local start = self.count + 1
      self.count = self.count + self.batchSize

      local idcs
      if self.count <= self.epochSize then
         idcs = self.idcs[{{start, self.count}}]
      else
         if start <= self.epochSize then
            idcs = self.idcs[{{start, self.epochSize}}]
            self.count = self.batchSize - idcs:size(1)
         else
            self.count = self.batchSize
         end
         if self.shuffle then
            self.idcs = torch.randperm(self.epochSize)
         end
         if start <= self.epochSize then
            idcs = torch.cat(idcs, self.idcs[{{1, self.count}}], 1)
         else
            idcs = self.idcs[{{1, self.count}}]
         end
      end

      self.pool:addjob(
         function(idcs)
            return threadDataset:getBatch(idcs)
         end,
         function(_batch_)
            batch = _batch_
         end,
         idcs
      )
   end

   self.pool:dojob()
   return batch
end

function DataProvider:reset()
   self.pool:synchronize()
   self.count = 0
   if self.shuffle then
      self.idcs = torch.randperm(self.epochSize)
   end
end


local function svhn(opt)
   local function load(dataPath)
      local loaded = torch.load(dataPath, 'ascii')
      local data = loaded.X:transpose(3, 4)
      local labels = loaded.y[1]
      return data, labels
   end

   local trainData, trainLabels = load(
      paths.concat(opt.dataDir, 'train_32x32.t7')
   )
   local extraData, extraLabels = load(
      paths.concat(opt.dataDir, 'extra_32x32.t7')
   )
   local testData, testLabels = load(
      paths.concat(opt.dataDir, 'test_32x32.t7')
   )

   local mean = {109.8820, 109.7119, 113.8176}
   local std = {50.1148, 50.5717, 50.8523}

   local trainDataset = datasets.ImageDataset(
      torch.cat(trainData, extraData, 1), torch.cat(trainLabels, extraLabels), 3, 32, 32, mean, std,
      nil, nil, nil, nil
   )
   local trainDataProvider = torch.DataProvider(
      opt, trainDataset, 'train'
   )

   local testDataset = datasets.ImageDataset(
      testData, testLabels, 3, 32, 32, mean, std,
      nil, nil, nil, nil
   )
   local testDataProvider = torch.DataProvider(
      opt, testDataset, 'test'
   )

   return trainDataProvider, testDataProvider
end

local function cifar(opt)
   local cifar = torch.load(
      paths.concat(opt.dataDir, opt.dataset .. '.t7')
   )

   local mean = {125.3069, 122.9504, 113.8654}
   local std = {62.9932, 62.0887, 66.7049}

   local trainDataset = datasets.ImageDataset(
      cifar.train.data, cifar.train.labels, 3, 32, 32, mean, std,
      nil, true, nil, 4
   )
   local trainDataProvider = torch.DataProvider(
      opt, trainDataset, 'train'
   )

   local testDataset = datasets.ImageDataset(
      cifar.val.data, cifar.val.labels, 3, 32, 32, mean, std,
      nil, nil, nil, nil
   )
   local testDataProvider = torch.DataProvider(
      opt, testDataset, 'test'
   )

   return trainDataProvider, testDataProvider
end

local function ucf101(opt)
--[[   local function load(dataPath, labelPath)
      local dataFile = torch.DiskFile(dataPath, 'r', true)
      local data = string.split(dataFile:readString('*a'), '\n')
      dataFile:close()
      for i = 1, #data do
         data[i] = string.split(data[i], ',')
      end

      local labelFile = torch.DiskFile(labelPath, 'r', true)
      local labels = string.split(labelFile:readString('*a'), '\n')
      labelFile:close()
      for i = 1, #labels do
         labels[i] = tonumber(labels[i])
      end

      return data, labels
   end

   local trainData, trainLabels = load(
      paths.concat(opt.dataDir, 'ucf_data_train1'),
      paths.concat(opt.dataDir, 'ucf_label_train1')
   )
   local testData, testLabels = load(
      paths.concat(opt.dataDir, 'ucf_data_test1'),
      paths.concat(opt.dataDir, 'ucf_label_test1')
   )
]]
local dataset = torch.load(paths.concat(opt.dataDir, opt.dataset .. '.t7'))
local trainData, trainLabels, testData, testLabels = dataset.train.data, dataset.train.labels, dataset.test.data, dataset.test.labels

   local mean =  {0.485 * 255, 0.456 * 255, 0.406 * 255}
   local std = {0.229 * 255, 0.224 * 255, 0.225 * 255}
   local trainDataset = datasets.VideoDataset(
      trainData, trainLabels, 3, opt.clipSize or 1, 224, 224, mean, std,
      'corner', true, nil, nil, nil, nil
   )
   local trainDataProvider = torch.DataProvider(
      opt, trainDataset, 'train'
   )

   local testDataset = datasets.VideoDataset(
      testData, testLabels, 3, opt.clipSize or 1, 224, 224, mean, std,
      'center', false, nil, nil, nil, nil
   )
   local testDataProvider = torch.DataProvider(
      opt, testDataset, 'test'
   )
   return trainDataProvider, testDataProvider
end

   

local function getDataProvider(opt)
   if opt.dataset == 'svhn' then
      return svhn(opt)
   end

   if opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
      return cifar(opt)
   end

   if opt.dataset == 'ucf101' then
      return ucf101(opt)
   end
end

return getDataProvider
