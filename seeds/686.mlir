module {
  func.func @main(%arg0: tensor<44xi16>, %arg1: tensor<1xi16>, %arg2: tensor<96x78xi8>, %arg3: tensor<96x1xi8>, %arg4: tensor<99x38xf32>) -> (tensor<44xi16>, tensor<96x78xi8>, tensor<99x38xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<44xi16>, tensor<1xi16>) -> tensor<44xi16>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<96x78xi8>, tensor<96x1xi8>) -> tensor<96x78xi8>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<96x78xi8>, tensor<96x78xi8>) -> tensor<96x78xi8>
    %3 = tosa.tanh %arg4 : (tensor<99x38xf32>) -> tensor<99x38xf32>
    %4 = tosa.sub %3, %3 : (tensor<99x38xf32>, tensor<99x38xf32>) -> tensor<99x38xf32>
    return %0, %2, %4 : tensor<44xi16>, tensor<96x78xi8>, tensor<99x38xf32>
  }
}
