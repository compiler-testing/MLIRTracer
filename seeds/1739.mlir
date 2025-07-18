module {
  func.func @main(%arg0: tensor<29x83x98x32x94xf32>, %arg1: tensor<99xi8>) -> (tensor<29x83x98x32x94xi1>, tensor<1xi8>) {
    %0 = tosa.rsqrt %arg0 : (tensor<29x83x98x32x94xf32>) -> tensor<29x83x98x32x94xf32>
    %1 = tosa.greater %0, %0 : (tensor<29x83x98x32x94xf32>, tensor<29x83x98x32x94xf32>) -> tensor<29x83x98x32x94xi1>
    %2 = tosa.reduce_sum %arg1 {axis = 0 : i32} : (tensor<99xi8>) -> tensor<1xi8>
    return %1, %2 : tensor<29x83x98x32x94xi1>, tensor<1xi8>
  }
}
