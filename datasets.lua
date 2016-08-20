local image = require 'image'
local datasets = {}

local function len(input)
   if type(input) == 'table' then
      return #input
   else
      return input:size(1)
   end
end


local ImageDataset = torch.class('ImageDataset', datasets)

function ImageDataset:__init(data, labels, c, h, w, mean, std,
   crop, hFlip, resize, translate)
   if labels then
      assert(len(data) == len(labels))
   end
   self.data = data
   self.labels = labels
   self.c = c or 3
   self.h = h or 32
   self.w = w or self.h

   self.mean = mean
   self.std = std
   self.crop = crop
   self.hFlip = hFlip
   self.resize = resize
   self.translate = translate
end

function ImageDataset:size()
   return len(self.data)
end

function ImageDataset:getBatch(idcs, h, w)
   local h = h or self.h
   local w = w or self.w
   local b = idcs:size(1)

   local samples = {}
   for i = 1, b do
      local idx = idcs[i]
      local img = self.data[idx]
      if torch.type(img) == 'string' then
         samples[i] = self:transform(img, h, w)
      else
         samples[i] = self:transform(img:clone(), h, w)
      end
   end
   local siz = samples[1]:size()
   samples = torch.FloatTensor.cat(samples, 1):view(b, siz[1], siz[2], siz[3])

   local labels = nil
   if self.labels then
      labels = {}
      for i = 1, b do
         local idx = idcs[i]
         labels[i] = self.labels[idx]
      end
      labels = torch.FloatTensor(labels)
   end

   local batch = {samples = samples, labels = labels}
   return batch
end

function ImageDataset:transform(img, h, w)
   if torch.type(img) == 'string' then
      img = image.load(img, self.c, 'byte')
   end

   if self.hFlip and torch.rand(1)[1] > 0.5 then
      img = image.hflip(img)
   end

   local img = img:float()
   if self.mean then
      if torch.type(self.mean) == 'number' then
         img = img - self.mean
      else
         for i = 1, #self.mean do
             img[i] = img[i] - self.mean[i]
         end
      end
   end
   if self.std then
      if torch.type(self.std) == 'number' then
         img = img / self.std
      else
         for i = 1, #self.std do
            img[i] = img[i] / self.std[i]
         end
      end
   end

   local cropSize
   if self.crop then
      cropSize = {h, w}
   end
   if self.resize then
      local resize = self.resize
      if #resize == 2 then
         local r = resize[1] + (resize[2] - resize[1]) * torch.rand(1)[1]
         cropSize[1] = cropSize[1] * r
         cropSize[2] = cropSize[2] * r
      else
         cropSize[1] = cropSize[1] * (resize[1] + (resize[2] - resize[1]) * torch.rand(1)[1])
         cropSize[2] = cropSize[2] * (resize[3] + (resize[4] - resize[3]) * torch.rand(1)[1])
      end

      local siz = img:size()
      for i = 1, 2 do
         cropSize[i] = math.floor(cropSize[i])
         cropSize[i] = math.min(cropSize[i], siz[2], siz[3])
      end
   end

   if self.crop == 'corner' then
      local format = {'c', 'tl', 'tr', 'bl', 'br'}
      cropFormat = format[torch.IntTensor(1):random(1, 5)[1]]
      img = image.crop(img, cropFormat, cropSize[2], cropSize[1])
   elseif self.crop == 'center' then
      img = image.crop(img, 'c', cropSize[2], cropSize[1])
   end

   if img:size(2) ~= h or img:size(3) ~= w then
      img = image.scale(img, w, h)
   end

   if self.translate then
      local pad = self.translate
      local temp = torch.FloatTensor(self.c, h + 2 * pad, w + 2 * pad):zero()
      temp[{{}, {pad + 1, pad + h}, {pad + 1, pad + w}}] = img
      local hh = torch.IntTensor(1):random(1, 2 * pad + 1)[1]
      local ww = torch.IntTensor(1):random(1, 2 * pad + 1)[1]
      img = temp[{{}, {hh, hh + h - 1}, {ww, ww + w - 1}}]
   end

   return img
end


local VideoDataset, VideoParent = torch.class('VideoDataset', 'ImageDataset', datasets)

function VideoDataset:__init(data, labels, c, t, h, w, mean, std,
   crop, hFlip, resize, translate, tFlip, timeResize)
   VideoParent.__init(self, data, labels, c, h, w, mean, std,
      crop, hFlip, resize, translate)
   self.t = t or 32
   self.tFlip = tFlip
   self.timeResize = timeResize
end

function VideoDataset:getBatch(idcs, t, h, w)
   local t = t or self.t
   local h = h or self.h
   local w = w or self.w
   local b = idcs:size(1)

   local samples = {}
   for i = 1, b do
      local idx = idcs[i]

      --[[local data = {}
      for j = 1, self.data[idx][2] do
         data[j] = self.data[idx][1] .. string.format('image_%04d.jpg', j)
      end]]

      samples[i] = self:transform(self.data[idx], t, h, w)--self:transform(self.data[idx], t, h, w)
   end
   siz = samples[1]:size()
   samples = torch.FloatTensor.cat(samples, 1):view(b, siz[1], siz[2], siz[3], siz[4])

   local labels = nil
   if self.labels then
      labels = {}
      for i = 1, b do
         local idx = idcs[i]
         labels[i] = self.labels[idx]
      end
   end
   labels = torch.FloatTensor(labels)

   local batch = {samples = samples, labels = labels}
   return batch
end

function VideoDataset:transform(data, t, h, w)
   local clipSize = t
   if self.timeResize then
      clipSize = torch.IntTensor(1):random(self.timeResize[1], self.timeResize[2])[1]
      --clipSize = clipSize * (self.timeResize[1] + (self.timeResize[2] - self.timeResize[1]) * torch.rand(1)[1])
      --clipSize = math.floor(clipSize)
   end

local interval = 1
local len = 1 + (clipSize - 1) * interval
local imgs = {}
if len >= data[2] then
   for i = 1, data[2], interval do
      imgs[i] = image.load(data[1] .. string.format('image_%04d.jpg', i), self.c, 'byte')
   end
else
   local start = torch.IntTensor(1):random(0, data[2] - len)[1]
   for i = 1, len, interval do
      imgs[i] = image.load(data[1] .. string.format('image_%04d.jpg', start + i), self.c, 'byte')
   end
end

   --[[local imgs = {}
   if clipSize >= data[2] then
      for i = 1, data[2] do
         imgs[i] = image.load(data[1] .. string.format('image_%04d.jpg', i), self.c, 'byte')
      end
   else
      local start = torch.IntTensor(1):random(0, data[2] - clipSize)[1]
      for i = 1, clipSize do
         imgs[i] = image.load(data[1] .. string.format('image_%04d.jpg', start + i), self.c, 'byte')
      end
   end]]

   local hFlip = false
   if self.hFlip and torch.rand(1)[1] > 0.5 then
      hFlip = true
   end

   local tFlip = false
   if self.tFlip and torch.rand(1)[1] > 0.5 then
      tFlip = true
   end

   local cropSize
   if self.crop then
      cropSize = {h, w}
   end
   if self.resize then
      local resize = self.resize
      local rand = torch.IntTensor(1)
      cropSize[1] = self.resize[rand:random(1, #self.resize)[1]]
      cropSize[2] = self.resize[rand:random(1, #self.resize)[1]]
      --[[local resize = self.resize
      if #resize == 2 then
         local r = resize[1] + (resize[2] - resize[1]) * torch.rand(1)[1]
         cropSize[1] = cropSize[1] * r
         cropSize[2] = cropSize[2] * r
      else
         cropSize[1] = cropSize[1] * (resize[1] + (resize[2] - resize[1]) * torch.rand(1)[1])
         cropSize[2] = cropSize[2] * (resize[3] + (resize[4] - resize[3]) * torch.rand(1)[1])
      end

      local siz = imgs[1]:size()
      for i = 1, 2 do
         cropSize[i] = math.floor(cropSize[i])
         cropSize[i] = math.min(cropSize[i], siz[2], siz[3])
      end]]
   end

   local cropFormat
   if self.crop == 'corner' then
      local format = {'c', 'tl', 'tr', 'bl', 'br'}
      cropFormat = format[torch.IntTensor(1):random(1, 5)[1]]
   elseif self.crop == 'center' then
      cropFormat = 'c'
   end

   local hh, ww
   if self.translate then
      hh = torch.IntTensor(1):random(1, 2 * self.translate + 1)[1]
      ww = torch.IntTensor(1):random(1, 2 * self.translate + 1)[1]
   end

   local clip = {}
   for i = 1, #imgs do
      local idx = i
      if tFlip then
         idx = #imgs + 1 - i
      end
      local img = imgs[idx]

      if hFlip then
         img = image.hflip(img)
      end

      if self.crop == 'corner' or self.crop == 'center' then
         img = image.crop(img, cropFormat, cropSize[2], cropSize[1])
      end

      if img:size(2) ~= h or img:size(3) ~= w then
         img = image.scale(img, w, h)
      end

      img = img:float()
      if self.mean then
         if torch.type(self.mean) == 'number' then
            img = img - self.mean
         else
            for j = 1, #self.mean do
               img[j] = img[j] - self.mean[j]
            end
         end
      end
      if self.std then
         if torch.type(self.std) == 'number' then
            img = img / self.std
         else
            for j = 1, #self.std do
               img[j] = img[j] / self.std[j]
            end
         end
      end

      if self.translate then
         local pad = self.translate
         local temp = torch.FloatTensor(self.c, h + 2 * pad, w + 2 * pad):zero()
         temp[{{}, {pad + 1, pad + h}, {pad + 1, pad + w}}] = img
         img = temp[{{}, {hh, hh + h - 1}, {ww, ww + w - 1}}]
      end

      clip[i] = img
   end

   if #clip < clipSize then
      local leftPad = math.floor((clipSize - #clip) / 2)
      local rightPad = clipSize - #clip - leftPad
      for i = 1, leftPad do
         table.insert(clip, 1, clip[1])--torch.FloatTensor(self.c, h, w):zero())
      end
      for i = 1, rightPad do
         table.insert(clip, #clip + 1, clip[#clip])--torch.FloatTensor(self.c, h, w):zero())
      end
   end

   local sample = clip
   if #clip ~= t then
      sample = {}
      local ratio = #clip / t
      for i = 1, t do
         local x = i * ratio
         
         if x <= 1 then
            sample[i] = clip[1]
         else
            local _, mod = math.modf(x, 1)
            if mod == 0 then
               sample[i] = clip[x]
            else
               local x1 = math.floor(x)
               local x2 = x1 + 1
               sample[i] = torch.mul(clip[x1], x2 - x) + torch.mul(clip[x2], x - x1)
            end
         end
      end
   end

   for i = 1, #sample do
      sample[i] = sample[i]:reshape(self.c, 1, h, w)
   end

   sample = torch.FloatTensor.cat(sample, 2)
   collectgarbage()
   return sample
end

return datasets
