module {
  func.func @main(%arg0: tensor<25x96xi1>) -> tensor<1x96xi1> {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<25x96xi1>) -> tensor<1x96xi1>
    %1 = tosa.logical_xor %0, %0 : (tensor<1x96xi1>, tensor<1x96xi1>) -> tensor<1x96xi1>
    return %1 : tensor<1x96xi1>
  }
}
