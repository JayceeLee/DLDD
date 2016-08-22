-----
-- Triplet Criterion which minimize the ratio between difference of positive pairs (should be small) and negative pair (should be big) 
-----


local TripletEmbeddingRatioCriterion, parent = torch.class('nn.TripletEmbeddingRatioCriterion', 'nn.Criterion')

function TripletEmbeddingRatioCriterion:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 0.01
   self.Li = torch.Tensor()
   self.gradInput = {}
end

function TripletEmbeddingRatioCriterion:updateOutput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)

   self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)) , torch.ones(N) - torch.cdiv((a - n):norm(2,2), (a - p):norm(2,2) + self.alpha), 2), 2)
   self.output = self.Li:sum() / N
   print(torch.cdiv((a - n):norm(2,2), (a - p):norm(2,2) + self.alpha))

   return self.output
end

function TripletEmbeddingRatioCriterion:updateGradInput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   self.gradInput = {}

   local margin = self.Li:gt(0):expand(N,a:size(2)):type(a:type()) 
   local norm_ap = (a-p):norm(2,2):expand(N,a:size(2)) 
   local norm_an = (a-n):norm(2,2):expand(N,a:size(2))

   self.gradInput[1] = -torch.cmul( torch.cdiv(a-n, torch.cmul(norm_an,norm_ap + self.alpha)) - torch.cdiv( torch.cmul(norm_an,a-p), torch.cmul(norm_ap,torch.pow(norm_ap + self.alpha, 2))),margin)/ N
   
   self.gradInput[2] = -torch.cmul(torch.cdiv( torch.cmul(norm_an, (a-p)), torch.cmul(norm_ap, torch.pow(norm_ap + self.alpha, 2))), margin)/ N
   
   self.gradInput[3] = torch.cmul(torch.cdiv(a-n, torch.cmul(norm_ap + self.alpha,norm_an)),margin)/ N

   self.gradInput = torch.concat({self.gradInput[1], self.gradInput[2], self.gradInput[3]}):view(3, N, self.gradInput[1]:size(2))

   return self.gradInput
end