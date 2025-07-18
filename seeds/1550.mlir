module {
  func.func @main(%arg0: tensor<52x31x66x4xi1>, %arg1: tensor<52x1x1x1xi1>) -> tensor<1x31x66x4xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<52x31x66x4xi1>, tensor<52x1x1x1xi1>) -> tensor<52x31x66x4xi1>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<52x31x66x4xi1>) -> tensor<1x31x66x4xi1>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<1x31x66x4xi1>) -> tensor<1x31x66x4xi1>
    return %2 : tensor<1x31x66x4xi1>
  }
}
