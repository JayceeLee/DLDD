
-- Loss implemented at A Discriminative Feature Learning Approach for Deep Face Recognition, Yandong Wen
-- Loss force to same class have similar representation 

local CenterCriterion, parent = torch.class('nn.CenterCriterion', 'nn.Criterion')

function CenterCriterion:__init(clusterCenters)
   parent.__init(self)
   self.clusterCenters = clusterCenters
   self.Li = torch.Tensor()
   self.gradInput = {}
end

function CenterCriterion:updateOutput(input, target)
   local N = input:size(1)
   self.Li  = torch.Tensor(input:size())
   for i=1,N do
     self.Li[i] = input[i] - self.clusterCenters[target[i]]
   end
   self.output = torch.pow(self.Li, 2):sum(2):sum()  / (2 * N)
  
   return self.output
end

function CenterCriterion:updateGradInput(input)
   local N = input:size(1)
   self.gradInput = self.Li / N
   return self.gradInput
end