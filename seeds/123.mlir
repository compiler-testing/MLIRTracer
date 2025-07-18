module {
  func.func @main(%arg0: tensor<75x40x19x56x5xi1>, %arg1: tensor<75x40x1x56x5xi1>, %arg2: tensor<51x61x92xi1>) -> (tensor<75x40x19x56x5xi1>, tensor<51x61x1xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<75x40x19x56x5xi1>, tensor<75x40x1x56x5xi1>) -> tensor<75x40x19x56x5xi1>
    %1 = tosa.reduce_all %arg2 {axis = 2 : i32} : (tensor<51x61x92xi1>) -> tensor<51x61x1xi1>
    return %0, %1 : tensor<75x40x19x56x5xi1>, tensor<51x61x1xi1>
  }
}
