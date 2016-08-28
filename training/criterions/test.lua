require 'nn'
require 'CenterCriterion'
require 'ContrastiveCriterion'
require 'GlobalCriterionTriplet'
require 'TripletEmbeddingRatioCriterion'
require 'TripletSimilarityCriterion'
require 'LiftedStructuredCritertion'
require 'MultiBatchCriterion'
require 'MagnetCriterion'
require 'TripletProbabilityCriterion'
require 'DistanceRatioCriterion'

local mytester = torch.Tester()
local precision = 1e-5
local expprecision = 1e-4

local function criterionJacobianTestTriplet(cri, input, idxTriplet)
   local eps = 1e-6
   local _ = cri:forward(input)
   local dfdx = cri:backward(input)
   
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx[idxTriplet])
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   local start, stop, diff = 1,input:nElement()/3, 0
   if idxTriplet == 2 then
     start = input:nElement()/3 + 1
     stop  = input:nElement()/3 * 2   
     diff =  input:nElement()/3 
   elseif idxTriplet == 3 then
     start = input:nElement()/3 * 2 + 1
     stop  = input:nElement()   
     diff =  input:nElement()/3  * 2
   end
   
   for i=start,stop do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i-diff] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx[idxTriplet]):abs():max()
   print(centraldiff_dfdx)
   print(dfdx[idxTriplet])
   mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
end

local function criterionJacobianTestPair(cri, input, target, idxPair)
   local eps = 1e-6
   local l = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx[idxPair])
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   local start, stop, diff = 1,input:nElement()/2, 0
   if idxPair == 2 then
     start = input:nElement()/2 + 1
     stop  = input:nElement()   
     diff =  input:nElement()/2 
   end 
   
   for i=start,stop do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i-diff] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx[idxPair]):abs():max()
   print(dfdx[idxPair])
   print(centraldiff_dfdx)
   mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
end

local function criterionJacobianTest(cri, input, target)
   local eps = 1e-1
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
--       print(fx1)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
--       print(fx2)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   print(dfdx)
   print(centraldiff_dfdx)
   mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
end


function CenterCriterion()
   local numLabels     = math.random(5,10)
   local input         = torch.rand(numLabels,10)
   local centerCluster = torch.rand(numLabels,10)
   local target        = torch.Tensor(numLabels)
   for i=1,numLabels do
      target[i] = math.random(1,numLabels)
   end
   
   local cri = nn.CenterCriterion()
   cri.centerCluster = centerCluster
   criterionJacobianTest(cri, input, target)
end

function ContrastiveCriterion()
   local numLabels     = math.random(5,10)
   local input         = torch.rand(2,numLabels,10)
   local target        = torch.Tensor(numLabels)
   for i=1,numLabels do
      local rand = math.random(-1,1)
      if rand == 0 then rand = rand - 1 end
      target[i] = rand
   end

   local cri = nn.ContrastiveCriterion(1.0)
   criterionJacobianTestPair(cri, input, target, 1)
   criterionJacobianTestPair(cri, input, target, 2)
end

function GlobalLoss()
  local numLabels     = 4 --math.random(5,10)
  local input         = torch.rand(3,numLabels,10)
  local cri = nn.GlobalCriterionTriplet()
  criterionJacobianTestTriplet(cri, input,3)
end

function TripletEmbeddingRatioCriterion()
  local numLabels     = math.random(5,10)
  local input         = torch.rand(3,numLabels,10)
  local cri = nn.TripletEmbeddingRatioCriterion()
  criterionJacobianTestTriplet(cri, input,1)
end

function TripletSimilarityCriterion()
  local numLabels     = math.random(5,10)
  local input         = torch.rand(3,numLabels,10)
  local cri = nn.TripletSimilarityCriterion()
  criterionJacobianTestTriplet(cri, input,2)
end

function LiftedStructuredCritertion()
  local numLabels     = math.random(5,10)
  local input         = torch.rand(3,numLabels,10)
  local cri = nn.LiftedStructuredCritertion()
  criterionJacobianTestTriplet(cri, input,3)
end

function MultiBatchCriterion()
   local numLabels     = math.random(5,10)
   local input         = torch.rand(2,numLabels,10)
   local target        = torch.Tensor(numLabels)
   for i=1,numLabels do
      target[i] = math.random(0,1) and 1 or -1 
   end

   local cri = nn.MultiBatchCriterion(1.0)
--    criterionJacobianTestPair(cri, input, target, 1)
   criterionJacobianTestPair(cri, input, target, 2)
end

function MagnetCriterion()
   local numLabels     = 3--math.random(5,10)
   local input         = torch.rand(numLabels,2)
   local centerCluster = torch.rand(numLabels,2)
   local target        = torch.Tensor(numLabels)
   for i=1,numLabels do
      target[i] = math.random(1,numLabels)
   end
   
   local cri = nn.MagnetCriterion(0.0)
   cri.centerCluster = centerCluster
   criterionJacobianTest(cri, input, target)
end

function TripletProbabilityCriterion()
  local numLabels     = math.random(5,10)
  local input         = torch.rand(3,numLabels,10)
  local cri = nn.TripletProbabilityCriterion()
  criterionJacobianTestTriplet(cri, input,3)
end

function DistanceRatioCriterion()
  local numLabels     = 3--math.random(5,10)
  local input         = torch.rand(3,numLabels,5)
  local cri = nn.DistanceRatioCriterion()
  criterionJacobianTestTriplet(cri, input,1)
end

-- CenterCriterion()
-- ContrastiveCriterion()
-- GlobalLoss()
-- TripletEmbeddingRatioCriterion()
-- TripletSimilarityCriterion()
-- LiftedStructuredCritertion()
-- MultiBatchCriterion()
-- MagnetCriterion()
-- TripletProbabilityCriterion()
DistanceRatioCriterion()

