-----
-- Learning a Metric Embedding for Face Recognition using the Multibatch Method, Oren Tadmor
-- Changes: This version does not use all possible negative pairs. It use samples made by PairSamplingModule
-----

require 'nn'
require 'torchx' --for concetration the table of tensors
require 'optim'
local MultiBatchCriterion, parent = torch.class('nn.MultiBatchCriterion', 'nn.Criterion')

function MultiBatchCriterion:__init(margin)
    parent.__init(self)
    self.margin = margin or 'auto'
    self.gradInput = {}
    self.epsilon = 0.0001
    if self.margin  == 'auto' then
      self.marginAuto = true
      self.margin = 1
      self.confusion = optim.ConfusionMatrix(2)
    end  
end

function MultiBatchCriterion:updateOutput(input, target)
    local input1, input2 = input[1], input[2] 
    local N              = input1:size(1)
    --calculate diff and dot product
    self.diff = input1 - input2
    local dot = torch.pow(self.diff,2):sum(2)
    
    self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(input1)),-torch.cmul(target,-dot + self.margin) + 1, 2) , 2)
    self.output = self.Li:sum() / (N)
    if self.marginAuto then
      self:updateMargin(dot, target)
    end
    return self.output
end


function MultiBatchCriterion:updateGradInput(input,target)
    local input1, input2 = input[1], input[2] 
    local N              = input1:size(1)
    local margin = self.Li:gt(0):type(input1:type()):cmul(target):expandAs(input1)
    
    self.gradInput[1] = torch.cmul(self.diff, margin) * 2 / N
    self.gradInput[2] = -torch.cmul(self.diff, margin) * 2 / N

    self.gradInput = torch.concat({self.gradInput[1], self.gradInput[2]}):view(2, N, self.gradInput[1]:size(2))
    
    if self.marginAuto then
      self.margin = self.newmargin
    end
    return self.gradInput
end

function MultiBatchCriterion:type(t)
    parent.type(self, t)
    return self
end

-- Find Marging with highest Accuracy
function MultiBatchCriterion:updateMargin(dot, target) 
  
  -- After having all embeding with pathes, we need to run verification process
  local threshold = torch.linspace(0,torch.max(dot),25) --from 0 to max, 25 elements
  local best_acc = 0
  local best_thres = 0
  for th=1,threshold:size()[1] do
    local thres = threshold[th]
    local acc = self:evalThresholdAccuracy(dot, target, thres)
    if acc > best_acc then
      best_thres = thres
      best_acc   = acc
    end
  end
  self.newmargin = best_thres
  print("M: " .. best_thres .. " " .. best_acc .. " " .. torch.max(dot))
end


function MultiBatchCriterion:evalThresholdAccuracy(distances, same_info, threshold )
  self.confusion:zero()
  for i=1,distances:size(1) do
    local diff = distances[i][1]
    if diff < threshold then
      self.confusion:add(1, same_info[i])
    else
      self.confusion:add(-1, same_info[i])
    end
  end
  self.confusion:updateValids()
  return self.confusion.totalValid
end 
