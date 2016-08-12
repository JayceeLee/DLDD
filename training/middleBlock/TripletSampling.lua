require 'torchx' --for concetration the table of tensors
local metrics = require 'metrics' --calculate Distance Matrix

local TripletSampling, parent = torch.class('nn.TripletSampling', 'nn.Module')


function TripletSampling:__init(opt, typeSampling)
   self.alpha          = opt.alpha -- margin within which the triplets schould be choosen
   self.peoplePerBatch = opt.peoplePerBatch
   self.embSize        = opt.embSize
   self.numPerClass    = {}
   self.sample         = self.sampleWithConstraints
   if typeSampling == 'random' then
    self.sample         = self.randomTriplets
   end
end

function TripletSampling:updateOutput(input)
  self.tripletIdx = {}
  return self:sample(input)
end

function TripletSampling:sampleWithConstraints(input)
  local as_table = {}
  local ps_table = {}
  local ns_table = {}

  local tripIdx     = 1
  local embStartIdx = 1
  local numTrips    = 0
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
          table.insert(ps_table,input[pIdx]:float())
          table.insert(ns_table,input[selNegIdx]:float())
          -- Add the original index of triplets.
          table.insert(self.tripletIdx, {aIdx,pIdx,selNegIdx})
          tripIdx = tripIdx + 1
        end
        numTrips = numTrips + 1
      end
      local aIdx = embStartIdx + n - 1
    end
    embStartIdx = embStartIdx + n
  end
  assert(embStartIdx - 1 == input:size(1))
  local nTripsFound = table.getn(as_table)
  print(('  + (nTrips, nTripsFound) = (%d, %d)'):format(numTrips, nTripsFound))
  if nTripsFound == 0 then
     print("Warning: nTripsFound == 0. Skipping batch.")
     self.output = torch.Tensor(3, 0, self.embSize)
     return self.output
  end

  self.output = self:concatOutput({as_table, ps_table, ns_table})
  return self.output
end

function TripletSampling:randomTriplets(input)
  local as_table = {}
  local ps_table = {}
  local ns_table = {}

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
	table.insert(as_table,input[aIdx]:float())
	table.insert(ps_table,input[pIdx]:float())
	table.insert(ns_table,input[selNegIdx]:float())
	-- Add the original index of triplets.
	table.insert(self.tripletIdx, {aIdx,pIdx,selNegIdx})
      end
      local aIdx = embStartIdx + n - 1
    end
    embStartIdx = embStartIdx + n
  end
  assert(embStartIdx - 1 == input:size(1))

  self.output = self:concatOutput({as_table, ps_table, ns_table})
  return self.output
end

function TripletSampling:concatOutput(dataTable)
  local nTripsFound = table.getn(dataTable[1])
  dataTable[1] = torch.concat(dataTable[1]):view(nTripsFound, self.embSize)
  dataTable[2] = torch.concat(dataTable[2]):view(nTripsFound, self.embSize)
  dataTable[3] = torch.concat(dataTable[3]):view(nTripsFound, self.embSize)
  return torch.concat(dataTable):view(3, nTripsFound, self.embSize)
end

function TripletSampling:updateGradInput(input, gradOutput)
   --map gradient to the index of input
  self.gradInput = torch.Tensor(input:size(1),input:size(2)):type(input:type())
  self.gradInput:zero()
  --get all gradient for each example
  for i=1,table.getn(self.tripletIdx) do
      self.gradInput[self.tripletIdx[i][1]]:add(gradOutput[1][i])
      self.gradInput[self.tripletIdx[i][2]]:add(gradOutput[2][i])
      self.gradInput[self.tripletIdx[i][3]]:add(gradOutput[3][i])
  end
  return self.gradInput
end


