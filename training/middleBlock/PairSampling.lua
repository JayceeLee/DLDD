require 'torchx' --for concetration the table of tensors
local metrics = require 'metrics' --calculate Distance Matrix

local PairSampling, parent = torch.class('nn.PairSampling', 'nn.Module')



function PairSampling:__init(opt, typeSampling)
   self.alpha          = opt.alpha -- margin within which the triplets schould be choosen
   self.peoplePerBatch = opt.peoplePerBatch
   self.embSize        = opt.embSize
   self.numPerClass    = {}
   self.sample         = self.sampleWithConstraints
   if typeSampling == 'random' then
    self.sample         = self.randomPairs
   end
   self.skipBatch       = false
end

function PairSampling:updateOutput(input)
  self.pairIdx = {}
  return self:sample(input)
end

function PairSampling:sampleWithConstraints(input)
  self.skipBatch = false
  local as_table = {}
  local sc_table = {}
  local target   = {}
  
  local pairIdx     = 1
  local embStartIdx = 1
  local numPair    = 0
  -- Calculate Distance Matrix and use it for choosing positive and Negative examples
  local distMat = metrics.distancesL2(input,input):float()
  for i = 1,self.peoplePerBatch do
    local n = self.numPerClass[i]
    for j = 1,n-1 do -- For every image in the batch.
      local aIdx = embStartIdx + j - 1
      local norms = distMat[{{aIdx}, {}}]:squeeze()
      for pair = j, n-1 do -- For every possible positive pair.
        local pIdx   = embStartIdx + pair 
        local normsP = norms - torch.Tensor(input:size(1)):fill(norms[pIdx])
        -- Set the indices of the same class to the max so they are ignored.
        normsP[{{embStartIdx,embStartIdx +n-1}}] = normsP:max()
        -- Get indices of images within the margin.
        local allNeg = normsP:lt(self.alpha):nonzero()
   
        -- Use only non-random triplets.
        -- Random triples (which are beyond the margin) will just produce gradient = 0,
        -- so the average gradient will decrease.
        if #allNeg:size() ~= 0 then
          selNegIdx = allNeg[math.random (allNeg:size(1))][1]
          -- Add the embeding of each example.
          table.insert(as_table,input[aIdx]:float())
          table.insert(sc_table,input[pIdx]:float())
	  table.insert(as_table,input[aIdx]:float())
          table.insert(sc_table,input[selNegIdx]:float())
           -- Add the original index of pairs.
	  table.insert(self.pairIdx, {aIdx,pIdx})
	  table.insert(self.pairIdx, {aIdx,selNegIdx})
	  -- Add target as positive and negative pair
	  table.insert(target, 2)
	  table.insert(target, 1)
          pairIdx = pairIdx + 1
        end
        numPair = numPair + 1
      end
      local aIdx = embStartIdx + n - 1
    end
    embStartIdx = embStartIdx + n
  end
  assert(embStartIdx - 1 == input:size(1))
  local nPairFound = table.getn(as_table)
  print(('  + (nTrips, nTripsFound) = (%d, %d)'):format(numPair, nPairFound))
  if nPairFound == 0 then
     print("Warning: nPairFound == 0. Skipping batch.")
     -- Set output of a=0, p=0,  |a-p| = 0
     self.output = torch.Tensor(2, 1, self.embSize):zero()
     table.insert(target, 2)
     self.skipBatch = true
     return self.output
  end

  self.output = self:concatOutput({as_table, ps_table, sc_table})
  return {self.output, torch.Tensor(target)}
end

function PairSampling:randomPairs(input)
  local as_table = {}
  local sc_table = {}
  local target   = {}

  local embStartIdx = 1
  for i = 1,self.peoplePerBatch do
    local n = self.numPerClass[i]
    -- get indexes of classes not in current class
    local negIdx = torch.cat(torch.range(1,embStartIdx), torch.range(embStartIdx + n - 1,input:size(1)))
    for j = 1,n-1 do -- For every image in the batch.
      local aIdx = embStartIdx + j - 1
      for pair = j, n-1 do -- For every possible positive pair.
        local pIdx   = embStartIdx + pair 
        
	selNegIdx = negIdx[math.random (negIdx:size(1))]
	-- Add the embeding of each example.
	table.insert(as_table,input[aIdx]:float()) -- postive pair
	table.insert(sc_table,input[pIdx]:float())
	table.insert(as_table,input[aIdx]:float()) -- negative pair
	table.insert(sc_table,input[selNegIdx]:float())
	-- Add the original index of pairs.
	table.insert(self.pairIdx, {aIdx,pIdx})
	table.insert(self.pairIdx, {aIdx,selNegIdx})
	-- Add target as positive and negative pair
	table.insert(target, 2)
	table.insert(target, 1)
      end
      local aIdx = embStartIdx + n - 1
    end
    embStartIdx = embStartIdx + n
  end
  assert(embStartIdx - 1 == input:size(1))

  self.output = self:concatOutput({as_table, sc_table})
  return {self.output, torch.Tensor(target)}
end

function PairSampling:concatOutput(dataTable)
  local nPairsFound = table.getn(dataTable[1])
  dataTable[1] = torch.concat(dataTable[1]):view(nPairsFound, self.embSize)
  dataTable[2] = torch.concat(dataTable[2]):view(nPairsFound, self.embSize)
  return torch.concat(dataTable):view(2, nPairsFound, self.embSize)
end

function PairSampling:updateGradInput(input, gradOutput)
   --map gradient to the index of input
  self.gradInput = torch.Tensor(input:size(1),input:size(2)):type(input:type())
  self.gradInput:zero()
  if self.skipBatch then -- if no triplet satisfy constrains, return zero gradient
    return self.gradInput
  end
  --get all gradient for each example
  for i=1,table.getn(self.pairIdx) do
      self.gradInput[self.pairIdx[i][1]]:add(gradOutput[1][i])
      self.gradInput[self.pairIdx[i][2]]:add(gradOutput[2][i])
  end
  -- Averege gradient per number of pairs?
  return self.gradInput
end


