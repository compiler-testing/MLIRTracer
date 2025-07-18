module {
  func.func @main(%arg0: tensor<49xi32>, %arg1: tensor<49xi32>, %arg2: tensor<47x83x94x28xi1>, %arg3: tensor<1x1x1x28xi1>) -> (tensor<49xi32>, tensor<47x83x94x28xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<49xi32>, tensor<49xi32>) -> tensor<49xi32>
    %1 = tosa.logical_or %arg2, %arg3 : (tensor<47x83x94x28xi1>, tensor<1x1x1x28xi1>) -> tensor<47x83x94x28xi1>
    return %0, %1 : tensor<49xi32>, tensor<47x83x94x28xi1>
  }
}
