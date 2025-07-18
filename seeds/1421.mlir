module {
  func.func @main(%arg0: tensor<66x51x79x84xi1>) -> tensor<66x1x79x84xi1> {
    %0 = tosa.reduce_any %arg0 {axis = 1 : i32} : (tensor<66x51x79x84xi1>) -> tensor<66x1x79x84xi1>
    %1 = tosa.logical_not %0 : (tensor<66x1x79x84xi1>) -> tensor<66x1x79x84xi1>
    return %1 : tensor<66x1x79x84xi1>
  }
}
