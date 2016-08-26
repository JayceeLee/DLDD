-----
-- Deep Metric Learning via Lifted Structured Feature Embedding, Hyun Oh Song
-- Changes: This version does not use all possible negative pairs. Just two negative per example (a-n,p-n)
-----
require 'torchx' --for concetration the table of tensors

local LiftedStructuredCritertion, parent = torch.class('nn.LiftedStructuredCritertion', 'nn.Criterion')

function LiftedStructuredCritertion:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 1.0
   self.Li = torch.Tensor()
   self.gradInput = {}
end

function LiftedStructuredCritertion:updateOutput(input)
   local a = input[1] -- anchor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   
   local posDiff =   (a - p):norm(2,2):pow(2)
   local negDiff_1 = (a - n):norm(2,2):pow(2)
   local negDiff_2 = (p - n):norm(2,2):pow(2)

   self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)), torch.log(torch.exp(-negDiff_1 + self.alpha ) + torch.exp(-negDiff_2 + self.alpha)) + posDiff, 2), 2)
   self.output = torch.pow(self.Li,2):sum() / (2*N)

   return self.output
end

function LiftedStructuredCritertion:updateGradInput(input)
   local a = input[1] -- anchor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   self.gradInput = {}
   
   local margin = self.Li:gt(0):type(a:type())
    
   local posDiff =   (a - p):norm(2,2):pow(2)
   local negDiff_1 = (a - n):norm(2,2):pow(2)
   local negDiff_2 = (p - n):norm(2,2):pow(2)
   local expNegDiff_1 = torch.exp(-negDiff_1 + self.alpha)
   local expNegDiff_2 = torch.exp(-negDiff_2 + self.alpha)

   self.gradInput[1] = self:normalize(torch.cmul(torch.cdiv(expNegDiff_1, expNegDiff_1 + expNegDiff_2):expandAs(a) ,n-a) + (a-p),N)
   self.gradInput[2] = self:normalize(torch.cmul(torch.cdiv(expNegDiff_2, expNegDiff_1 + expNegDiff_2):expandAs(a) ,n-p) + (p-a),N)
   self.gradInput[3] = self:normalize(torch.cdiv(((a-n):cmul(expNegDiff_1:expandAs(a)) + (p-n):cmul(expNegDiff_2:expandAs(a))), (expNegDiff_1 + expNegDiff_2):expandAs(a)),N)
   self.gradInput = torch.concat({self.gradInput[1], self.gradInput[2], self.gradInput[3]}):view(3, N, self.gradInput[1]:size(2))
   return self.gradInput
end

function LiftedStructuredCritertion:normalize(data, N)
  return torch.cmul(torch.cmul(data,self.Li:expandAs(data)), self.Li:gt(0):expandAs(data):type(data:type())) * 2 / N
end
