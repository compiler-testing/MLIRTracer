module {
  func.func @main(%arg0: tensor<78x45x56x78x97xi1>, %arg1: tensor<1x45x1x1x97xi1>) -> tensor<78x45x56x78x97xi1> {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<78x45x56x78x97xi1>, tensor<1x45x1x1x97xi1>) -> tensor<78x45x56x78x97xi1>
    %1 = tosa.bitwise_and %0, %0 : (tensor<78x45x56x78x97xi1>, tensor<78x45x56x78x97xi1>) -> tensor<78x45x56x78x97xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<78x45x56x78x97xi1>, tensor<78x45x56x78x97xi1>) -> tensor<78x45x56x78x97xi1>
    return %2 : tensor<78x45x56x78x97xi1>
  }
}
