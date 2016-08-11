-- Parallel Criterion which output error of each Criterion separately, not weighted
local EmptyCriterion, parent = torch.class('nn.EmptyCriterion', 'nn.Criterion')

function EmptyCriterion:__init()
   parent.__init(self)
end

function EmptyCriterion:updateOutput(input, target)
   self.output = 0
   return self.output
end

function EmptyCriterion:updateGradInput(input, target)
   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
   nn.utils.recursiveFill(self.gradInput, 0)
   return self.gradInput
end

function EmptyCriterion:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end