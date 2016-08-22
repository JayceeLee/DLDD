--------------------------------------------------------------------------------
-- TripletEmbeddingCriterion
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr/May 15
--------------------------------------------------------------------------------
require 'torchx' --for concetration the table of tensors

local TripletEmbeddingCriterion, parent = torch.class('nn.TripletEmbeddingCriterion', 'nn.Criterion')

function TripletEmbeddingCriterion:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 0.2
   self.Li = torch.Tensor()
   self.gradInput = {}
end

function TripletEmbeddingCriterion:updateOutput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   self.Li,_ = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)) , (a - p):norm(2,2):pow(2) -  (a - n):norm(2,2):pow(2) + self.alpha, 2), 2)
   self.output = self.Li:sum() / N
   return self.output
end

function TripletEmbeddingCriterion:updateGradInput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   self.gradInput = {}

   self.gradInput[1] = (n - p):cmul(self.Li:gt(0):repeatTensor(1, a:size(2)):type(a:type()) * 2/N)
   self.gradInput[2] = (p - a):cmul(self.Li:gt(0):repeatTensor(1, a:size(2)):type(a:type()) * 2/N)
   self.gradInput[3] = (a - n):cmul(self.Li:gt(0):repeatTensor(1, a:size(2)):type(a:type()) * 2/N)
   
   self.gradInput = torch.concat({self.gradInput[1], self.gradInput[2], self.gradInput[3]}):view(3, N, self.gradInput[1]:size(2))
   
   return self.gradInput
end
