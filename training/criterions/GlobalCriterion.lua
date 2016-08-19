--------------------------------------------------------------------------------
-- "Learning Local Image Descriptors with Deep Siamese and Triplet Convolutional Networks by Minimizing Global Loss Functions"
-- Vijay Kumar B G, Gustavo Carneiro, Ian Reid
-------------------------------------------------------------------------------

local GlobalCriterion, parent = torch.class('nn.GlobalCriterion', 'nn.Criterion')

function GlobalCriterion:__init(alpha, lambda)
   parent.__init(self)
   self.alpha = alpha or 0.4
   self.lambda = lambda or 0--0.8
   self.Li = torch.Tensor()
   self.gradInput = {}
end

function GlobalCriterion:updateOutput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)

   self.d_plus     = (a-p):pow(2):sum(2) / 4 
   self.d_minus    = (a-n):pow(2):sum(2) / 4
   self.mean_plus  = self.d_plus:sum() / N
   self.mean_minus = self.d_minus:sum() / N
   
   local var_plus  = torch.add(self.d_plus, -self.mean_plus):pow(2):sum()/N
   local var_minus = torch.add(self.d_minus, -self.mean_minus):pow(2):sum()/N
   
   self.output = (var_plus + var_minus) + self.lambda * math.max(0,self.mean_plus - self.mean_minus + self.alpha)
   return self.output
end

function GlobalCriterion:updateGradInput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   self.gradInput = {}

   self.gradInput[1] = 2*torch.add(self.d_plus, -self.mean_plus):expand(N,a:size(2)):cmul(p - a) + 2*torch.add(self.d_minus, -self.mean_minus):expand(N,a:size(2)):cmul(n-a) + (p-n):mul(self.lambda*math.max(0, self.mean_minus -  self.mean_plus - self.alpha))/(-2*N)
   
   self.gradInput[2] = 2*torch.add(self.d_plus, -self.mean_plus):expand(N,a:size(2)):cmul(a-p) +  (a-p):mul(self.lambda * math.max(0, self.mean_minus -  self.mean_plus - self.alpha))/ (-2*N)
   
   self.gradInput[3] =2*torch.add(self.d_minus, -self.mean_minus):expand(N,a:size(2)):cmul(a-n) + (n-a):mul(self.lambda *math.max(0, self.mean_minus -  self.mean_plus - self.alpha)) /(-2*N)
   
   self.gradInput = torch.concat({self.gradInput[1], self.gradInput[2], self.gradInput[3]}):view(3, N, self.gradInput[1]:size(2))

   return self.gradInput
end
