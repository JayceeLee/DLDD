
local ImageData = require 'datasets/imageData'
local FileData = torch.class('dldd.FileData', ImageData)

function FileData:__init(imageInfo, opt,split)
   self.imageInfo = imageInfo[split]
   self.classList = imageInfo.classList
   self.split = split
   self.opt   = opt
   self.imgDim = opt.imgDim
   self.size   = self.imageInfo.labels:size()
   self:createClassSample()
   self.indexes = torch.range(1, self.size)
end

-- For sampling specific classes, we need a information of idx of each class
function FileData:createClassSample()
    self.imageInfo.classListSample  = {}
    for idx=1,torch.max(self.imageInfo.labels) do
      classListSample[idx] = {}
    end
    for idx=1,self.imageInfo.labels:size() do
      table.insert(classListSample[self.imageInfo.labels[idx]], idx)
    end
    
    for idx=1,torch.max(self.imageInfo.labels) do
      classListSample[idx] = torch.LongTensor(classListSample[idx])
    end
end

function FileData:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

return ImageData.FileData
