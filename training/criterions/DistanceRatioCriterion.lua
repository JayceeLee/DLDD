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
    local a = input[1] -- ancor
    local p = input[2] -- positive
    local n = input[3] -- negative
    local N = a:size(1)
    
    -- Get difference between positive pair and min of negative pair
    self.inputDiff = torch.concat({(a - p):norm(2,2):pow(2), torch.min((a - n):norm(2,2):pow(2), (p - n):norm(2,2):pow(2),2)}):view(2, N, a:size(2))
    if not self.Target:isSameSizeAs(self.inputDiff) then
        self:createTarget(self.inputDiff, 1)
    end
    self.output = self.MSE:updateOutput(self.SoftMax:updateOutput(self.inputDiff),self.Target)
    return self.output
end

function DistanceRatioCriterion:updateGradInput(input)
    if not self.Target:isSameSizeAs(input) then
        self:createTarget(input, 1)
    end

    self.gradInput = self.SoftMax:updateGradInput(self.inputDiff, self.MSE:updateGradInput(self.SoftMax.output,self.Target))
    --
    return self.gradInput
end

function DistanceRatioCriterion:type(t)
    parent.type(self, t)
    self.SoftMax:type(t)
    self.MSE:type(t)
    self.Target = self.Target:type(t)
    return self
end