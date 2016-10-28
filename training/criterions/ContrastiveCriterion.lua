require 'nn'
require 'torchx' --for concetration the table of tensors
require 'optim'
local ContrastiveCriterion, parent = torch.class('nn.ContrastiveCriterion', 'nn.Criterion')

function ContrastiveCriterion:__init(margin)
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

-- Loss = 0.5 * Y * ||D1 -D2||**2 + 0.5 * (1 - Y)  * max(0, margin -||D1 -D2|| ) ** 2
function ContrastiveCriterion:updateOutput(input, target)
    local input1, input2 = input[1], input[2] 
    local N              = input1:size(1)
    --calculate diff and dot product
    self.diff = input1 - input2
    local dot = torch.pow(self.diff,2):sum(2)
    self.Li  = torch.Tensor(N):type(input:type())
    self.negative  = torch.Tensor(N):type(input:type())
    self.diff_margin  = torch.Tensor(N):type(input:type())
    
    for i=1,N do
      if target[i] == 1 then  --select positive example
         self.Li[i] =  dot[i]
      else --calc negative loss
         self.negative[i]    =  torch.sqrt(dot[i])
         self.diff_margin[i] = self.margin - self.negative[i]
         self.Li[i]          = math.pow(math.max(self.diff_margin[i], 0.0),2)
      end
    end
    
    self.output = self.Li:sum() / (2 * N)
    if self.marginAuto then
      self:updateMargin(dot, target)
    end
    return self.output
end

--[[
        gradient: 
       positive: 
            D1 -D2 (in case of D1) 
           -(D1 - D2) (in case of D2)
       negative: if (margin -(D1 -D2)) > 0, then 
          (margin - ||D1 - D2||)/||D1 - D2|| *(D1 - D2) * (-1)  (in case of D1);
                    (margin - ||D1 - D2||)/||D1 - D2|| *(D1 - D2) * (1)   (in case of D2)
                 else: 
          0 

--]]
function ContrastiveCriterion:updateGradInput(input,target)
    local input1, input2 = input[1], input[2] 
    local N              = input1:size(1)
    self.gradInput    = {}
    self.gradInput[1] = torch.Tensor():resizeAs(input1):typeAs(input1):zero()
    self.gradInput[2] = torch.Tensor():resizeAs(input1):typeAs(input1):zero()

    for i=1,N do
      if target[i] == 1 then
         self.gradInput[1][i] = self.diff[i]
         self.gradInput[2][i] = - self.diff[i]
      else
         if self.diff_margin[i] > 0 then -- if example are too close to each other, get gradient. Otherwise, set zeros
           local alpha = ((self.margin - self.negative[i])/(self.negative[i] + self.epsilon)) * self.diff[i]
           self.gradInput[1][i]  = - alpha
           self.gradInput[2][i]  =   alpha
         end
      end
    end

    self.gradInput = torch.concat({self.gradInput[1], self.gradInput[2]}):view(2, N, self.gradInput[1]:size(2))
    self.gradInput = self.gradInput / N
    
    if self.marginAuto then
      self.margin = self.newmargin
    end
    return self.gradInput
end

function ContrastiveCriterion:type(t)
    parent.type(self, t)
    return self
end

-- Find Marging with highest Accuracy
function ContrastiveCriterion:updateMargin(dot, target) 
  
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


function ContrastiveCriterion:evalThresholdAccuracy(distances, same_info, threshold )
  self.confusion:zero()
  for i=1,distances:size(1) do
    local diff = distances[i][1]
    if diff < threshold then
      self.confusion:add(2, same_info[i] > 0 and 2 or 1)
    else
      self.confusion:add(1, same_info[i] > 0 and 2 or 1)
    end
  end
  self.confusion:updateValids()
  return self.confusion.totalValid
end 
