module {
  func.func @main(%arg0: tensor<11x56x53x58x74xi32>, %arg1: tensor<11x1x53x58x1xi32>, %arg2: tensor<82xf32>) -> (tensor<11x56x53x58x74xi32>, tensor<1xf32>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<11x56x53x58x74xi32>, tensor<11x1x53x58x1xi32>) -> tensor<11x56x53x58x74xi32>
    %1 = tosa.reduce_sum %arg2 {axis = 0 : i32} : (tensor<82xf32>) -> tensor<1xf32>
    return %0, %1 : tensor<11x56x53x58x74xi32>, tensor<1xf32>
  }
}
