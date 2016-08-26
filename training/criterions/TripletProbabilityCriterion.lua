--------------------------------------------------------------------------------
-- TRIPLET SIMILARITY EMBEDDING FOR FACE VERIFICATION, Swami Sankaranarayanan
--------------------------------------------------------------------------------

require 'torchx' --for concetration the table of tensors

local TripletProbabilityCriterion, parent = torch.class('nn.TripletProbabilityCriterion', 'nn.Criterion')

function TripletProbabilityCriterion:__init()
   parent.__init(self)
   self.Li = torch.Tensor()
   self.gradInput = {}
end

function TripletProbabilityCriterion:updateOutput(input)
   local a = input[1] -- anchor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   
   self.diffPos = torch.exp(torch.cmul(a,p):sum(2))
   self.diffNeg= torch.exp(torch.cmul(a,n):sum(2))
   
   self.Li = - torch.log(torch.cdiv(self.diffPos, self.diffPos + self.diffNeg))
   self.output = self.Li:sum() / N
   return self.output
end

function TripletProbabilityCriterion:updateGradInput(input)
   local a = input[1] -- anchor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   self.gradInput = {}
   self.gradInput[1] = torch.cdiv(torch.cmul(self.diffPos:expandAs(a),p) + torch.cmul(self.diffNeg:expandAs(a),n),(self.diffPos +  self.diffNeg):expandAs(a)) - p
   self.gradInput[2] = torch.cdiv(torch.cmul(self.diffPos:expandAs(a),a) ,(self.diffPos +  self.diffNeg):expandAs(a)) - a
   self.gradInput[3] = torch.cdiv(torch.cmul(self.diffNeg:expandAs(a),a) ,(self.diffPos +  self.diffNeg):expandAs(a))
   
   self.gradInput = torch.concat({self.gradInput[1], self.gradInput[2], self.gradInput[3]}):view(3, N, self.gradInput[1]:size(2))/N
   
   return self.gradInput
end