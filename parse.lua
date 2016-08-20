local function parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option('-dataset',        'none',        'svhn | cifar10 | siftflow | ucf101')
   cmd:option('-model',          'none',        'Model definition file')
   cmd:option('-saveDir',        'none',        'Path to save the results')

   cmd:option('-lr',             0.1,           'Initial learning rate')
   cmd:option('-nGPU',           4,             'Number of GPUs')
   cmd:option('-chunkSize',      'none',        'Size of chunk')
   cmd:option('-nChunks',        100,           'Number of total chunks')
   cmd:option('-batchSize',      256,           'Size of mini-batch')
   cmd:option('-dataDir',        'none',        'Path to dataset')
   cmd:option('-nClasses',       10,            'Number of classes in the dataset')
   cmd:option('-nThreads',       4,             'Number of data provider threads')
   cmd:option('-shuffle',        'true',        'Whether shuffle the samples in each epoch')
   
   cmd:option('-chunk',          1,             'Number of chunks that have gone')
   cmd:option('-testOnly',       'false',       'Only test on the entire test set')
   cmd:option('-seed',           0,             'RNG seed')
   cmd:option('-momentum',       0.9,           'momentum')
   cmd:option('-weightDecay',    1e-4,          'weight decay')
   cmd:option('-resume',         'none',        'Path to resume file')
   cmd:option('-loadModel',      'none',        'Path to pretrained Model')

   cmd:option('-clipSize',       1,             'Number of frames')

   cmd:text()

   local opt = cmd:parse(arg or {})

   if opt.saveDir == 'none' then
      opt.saveDir = './tempSaveDir'
   end
   if opt.chunkSize == 'none' then
      opt.chunkSize = nil
   end
   if opt.dataDir == 'none' then
      opt.dataDir = './data/' .. opt.dataset
   end
   opt.shuffle = opt.shuffle ~= 'false'
   opt.testOnly = opt.testOnly ~= 'false'
   if opt.resume == 'none' then
      opt.resume = nil
   end
   if opt.loadModel == 'none' then
      opt.loadModel = nil
   end

   return opt
end

return parse
