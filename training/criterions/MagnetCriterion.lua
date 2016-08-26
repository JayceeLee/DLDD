--------------------------------------------------------------------------------
-- "METRIC LEARNING WITH ADAPTIVE DENSITY DISCRIMINATION", Oren Rippel
-- Note: Version does rely only on class clusters. Does not recomute any clusters inside clusters, like in paper
-- The cluster center are provided ahead, so we do not calculate it every iteration here
-------------------------------------------------------------------------------
local metrics = require 'metrics'
local MagnetCriterion, parent = torch.class('nn.MagnetCriterion', 'nn.Criterion')

function MagnetCriterion:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 1.0
   self.centerCluster = nila
   self.gradInput = {}
end


--[[ Steps:
    1. Calulate distance between input and all cluster centers 
    2. Collect distance between feature and assigned center and calculate variation
    3. Collect distance between feature and not-assigned center.
    4. Calucalate per example loss using max-margin function
--]]
function MagnetCriterion:updateOutput(input, target)
   local N = input:size(1)
   
   local distMat = metrics.distancesL2(input,self.centerCluster):pow(2)
   self.distCenterL2  = torch.Tensor(N)
   for i=1,N do
     self.distCenterL2[i] = distMat[{{i},{target[i]}}]
   end
   
   self.variation = self.distCenterL2:sum()/(N-1)
--    local diffBadCenter = torch.exp(distMat / (-2*self.variation))
--    for i=1,N do -- set 0.0 for diff from target center
--      diffBadCenter[{{i},{target[i]}}] = 0.0
--    end
--    local  diffBadCenterSum = 1--diffBadCenter:sum(2)
   self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(input)),self.distCenterL2/(2*self.variation) + self.alpha,2),2)
   
--       self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(input)),self.distCenterL2/(2*self.variation) + self.alpha + torch.log(diffBadCenterSum),2),2)
      
   self.output =  self.Li:sum()/N
   return self.output
end

function MagnetCriterion:updateGradInput(input, target)
   local N = input:size(1) 
--    local margin = self.Li:gt(0):type(input:type()):expandAs(input)
--    local diffCenter  = torch.Tensor(input:size())
--    for i=1,N do
--      diffCenter[i] = input[i] - self.centerCluster[target[i]]
--    end
--    self.gradInput = torch.cmul(diffCenter/(self.variation) - (self.distCenterL2 * torch.pow(self.variation,3/2)):view(N,1):expandAs(input), margin)
--    self.gradInput = self.gradInput + torch.cmul(diffCenter/(self.variation)
   return self.gradInput
end
