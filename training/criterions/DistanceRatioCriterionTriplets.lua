-- Taken from Elad Hoffer's TripletNet https://github.com/eladhoffer 
-- Hinge loss ranking could also be used, see below
-- https://github.com/torch/nn/blob/master/doc/criterion.md#nn.MarginRankingCriterion
require 'nn'	
require 'torchx' --for concetration the table of tensors
local DistanceRatioCriterion, parent = torch.class('nn.DistanceRatioCriterion', 'nn.Criterion')

function DistanceRatioCriterion:__init()
    parent.__init(self)
    self.SoftMax = nn.SoftMax()
    self.MSE = nn.MSECriterion()
    self.Target = torch.Tensor()
end

function DistanceRatioCriterion:createTarget(input, target)
    local target = target or 1
    self.Target:resizeAs(input):typeAs(input):zero()
    self.Target[{{},target}]:add(1)
end

function DistanceRatioCriterion:updateOutput(input)
    local a = input[1] -- anchor
    local p = input[2] -- positive
    local n = input[3] -- negative
    local N = a:size(1)
    
    -- Get difference between positive pair and min of negative pair
--     local hardTriplet, Idx = torch.min(torch.concat({(a - n):norm(2,2):pow(2), (p - n):norm(2,2):pow(2)},2):view(N,2),2)
--     self.Idx = Idx
    self.inputDiff = torch.concat({(a - p):pow(2):sum(2), (p - n):pow(2):sum(2)}):view(N,2)
    if not self.Target:isSameSizeAs(self.inputDiff) then self:createTarget(self.inputDiff, 1) end
    
    self.output = self.MSE:updateOutput(self.SoftMax:updateOutput(self.inputDiff),self.Target)
    return self.output
end

function DistanceRatioCriterion:updateGradInput(input)
    local a = input[1] -- anchor
    local p = input[2] -- positive
    local n = input[3] -- negative
    local N = a:size(1)
    if not self.Target:isSameSizeAs(self.inputDiff) then  self:createTarget(self.inputDiff, 1) end
     
    -- Get which negative tripelet was smaller
--     print(self.Idx)
--     local firstTriplet = self.Idx:clone():expandAs(a):double()
--     firstTriplet[torch.eq(firstTriplet,1)] = 0
--     local secondTriplet = self.Idx:clone():expandAs(a):double()
--     secondTriplet = secondTriplet - 1
    
    self.gradInput = {}
    local gradInputMSE = self.SoftMax:updateGradInput(self.inputDiff, self.MSE:updateGradInput(self.SoftMax.output,self.Target))
    local ap = gradInputMSE[{{},{1}}]:expandAs(a)
    local hard = gradInputMSE[{{},{2}}]:expandAs(a)
    
    print(ap)
    print(a-p)

    self.gradInput[1] = (a-p):cmul(ap) * 2/N
    self.gradInput[2] = (a-p):cmul(ap)
    self.gradInput[3] = (a-p):cmul(ap)
        
    self.gradInput = torch.concat({self.gradInput[1], self.gradInput[2], self.gradInput[3]}):view(3, N, self.gradInput[1]:size(2))
    
    return self.gradInput
end

function DistanceRatioCriterion:type(t)
    parent.type(self, t)
    self.SoftMax:type(t)
    self.MSE:type(t)
    self.Target = self.Target:type(t)
    return self
end