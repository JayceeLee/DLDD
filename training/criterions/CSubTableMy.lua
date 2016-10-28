local CSubTableMy, parent = torch.class('nn.CSubTableMy', 'nn.Module')

function CSubTableMy:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CSubTableMy:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   self.output:add(-1,input[2])  
   return self.output
end

function CSubTableMy:updateGradInput(input, gradOutput)
   input = input:cuda()
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[1].new()
   self.gradInput[1]:resizeAs(input[1]):copy(gradOutput)
   self.gradInput[2]:resizeAs(input[2]):copy(gradOutput):mul(-1)

   return self.gradInput
end