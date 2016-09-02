--------------------------------------------------------------------------------
-- TRIPLET SIMILARITY EMBEDDING FOR FACE VERIFICATION, Swami Sankaranarayanan
--------------------------------------------------------------------------------

require 'torchx' --for concetration the table of tensors

local TripletSimilarityCriterion, parent = torch.class('nn.TripletSimilarityCriterion', 'nn.Criterion')

function TripletSimilarityCriterion:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 0.1
   self.Li = torch.Tensor()
   self.gradInput = {}
end

function TripletSimilarityCriterion:updateOutput(input)
   local a = input[1] -- anchor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)), torch.cmul(a,n):sum(2) -  torch.cmul(a,p):sum(2) + self.alpha, 2), 2)
   self.output = self.Li:sum() / N
   return self.output
end

function TripletSimilarityCriterion:updateGradInput(input)
   local a = input[1] -- anchor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   local dataZero = self.Li:gt(0):repeatTensor(1, a:size(2)):type(a:type()) * 1/N
   self.gradInput = {}
   self.gradInput[1] = torch.cmul(n - p, dataZero)
   self.gradInput[2] = torch.cmul(-a, dataZero)
   self.gradInput[3] = torch.cmul(a, dataZero)
   
   self.gradInput = torch.concat({self.gradInput[1], self.gradInput[2], self.gradInput[3]}):view(3, N, self.gradInput[1]:size(2))
   
   return self.gradInput
end