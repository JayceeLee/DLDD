-- Parallel Criterion which output error of each Criterion merged and repeat input
local ParallelCriterionMerge, parent = torch.class('nn.ParallelCriterionMerge', 'nn.Criterion')

function ParallelCriterionMerge:__init(repeatTarget, repeatInput)
   parent.__init(self)
   self.criterions = {}
   self.weights = {}
   self.gradInput = {}
   self.repeatTarget = repeatTarget
   self.repeatInput = repeatInput
end

function ParallelCriterionMerge:add(criterion, weight)
   assert(criterion, 'no criterion provided')
   weight = weight or 1
   table.insert(self.criterions, criterion)
   table.insert(self.weights, weight)
   return self
end

function ParallelCriterionMerge:updateOutput(input, target)
   self.output = {}
   for i,criterion in ipairs(self.criterions) do
      local target = self.repeatTarget and target or target[i]
      local input  = self.repeatInput and input or input[i]
      table.insert(self.output, criterion:updateOutput(input,target))
   end
   return self.output
end

function ParallelCriterionMerge:updateGradInput(input, target)
   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
   nn.utils.recursiveFill(self.gradInput, 0)
   for i,criterion in ipairs(self.criterions) do
      local target = self.repeatTarget and target or target[i]
      local input  = self.repeatInput and input or input[i]
      nn.utils.recursiveAdd(self.gradInput, self.weights[i], criterion:updateGradInput(input, target))
   end
   return self.gradInput
end

function ParallelCriterionMerge:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end