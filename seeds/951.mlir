module {
  func.func @main(%arg0: tensor<45x41x70x68x81xi1>, %arg1: tensor<45x41x70x68x81xi1>, %arg2: tensor<32x89x85xi1>) -> (tensor<45x41x70x68x81xi1>, tensor<1x89x85xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<45x41x70x68x81xi1>, tensor<45x41x70x68x81xi1>) -> tensor<45x41x70x68x81xi1>
    %1 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<32x89x85xi1>) -> tensor<1x89x85xi1>
    return %0, %1 : tensor<45x41x70x68x81xi1>, tensor<1x89x85xi1>
  }
}
