require 'nn'
require 'cunn'
require 'cudnn'

-- Add supported rnns, and their gate sizes
local supported_rnns = {}
supported_rnns['cudnn.RNNReLU'] = 1
supported_rnns['cudnn.RNNTanh'] = 1
supported_rnns['cudnn.GRU'] = 3
supported_rnns['cudnn.BGRU'] = 3
supported_rnns['cudnn.LSTM'] = 4
supported_rnns['cudnn.BLSTM'] = 4

local cmd = torch.CmdLine()
cmd:text('Processes RNNs before converting to pytorch')
cmd:option('-model', '', 'Path to the input model')
cmd:option('-output', '', 'Path to the processed output model')

local opt = cmd:parse(arg)

local model = torch.load(opt.model)
local outputPath = opt.output

local function process_rnn(rnn, gateSize)
    local inputSize = rnn.inputSize
    local hiddenSize = rnn.hiddenSize
    local directions = rnn.numDirections
    local numLayers = rnn.numLayers
    local weights = rnn:weights()
    local biases = rnn:biases()
    -- store for weights in cuDNN format
    rnn.all_weights = {}
    for nLayer = 1, numLayers / directions do
        for direction = 0, directions - 1 do
            local layerInputSize = nLayer == 1 and inputSize or hiddenSize * directions
            local wi = torch.FloatTensor(gateSize, layerInputSize * hiddenSize)
            local wh = torch.FloatTensor(gateSize, hiddenSize * hiddenSize)
            local bi = torch.FloatTensor(gateSize, hiddenSize)
            local bh = torch.FloatTensor(gateSize, hiddenSize)

            local layerWeights = weights[nLayer + direction]
            local layerBiases = biases[nLayer + direction]
            for x = 1, gateSize do
                wi[x]:copy(layerWeights[x])
                wh[x]:copy(layerWeights[x + gateSize])
                bi[x]:copy(layerBiases[x])
                bh[x]:copy(layerBiases[x + gateSize])
            end
            rnn.all_weights[#rnn.all_weights] = { wi, wh, bi, bh }
        end
    end
    return rnn
end

for moduleName, gateSize in pairs(supported_rnns) do
    local nodes, container_nodes = model:findModules(moduleName)
    for i = 1, #nodes do
        -- Search the container for the current node
        for j = 1, #(container_nodes[i].modules) do
            if container_nodes[i].modules[j] == nodes[i] then
                local module = process_rnn(container_nodes[i].modules[j], gateSize)
                container_nodes[i].modules[j] = module
            end
        end
    end
end

model = model:float()

torch.save(outputPath, model)
