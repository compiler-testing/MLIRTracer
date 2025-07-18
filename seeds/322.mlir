module {
  func.func @main(%arg0: tensor<73x74xi1>, %arg1: tensor<1x74xi1>, %arg2: tensor<28xf32>) -> (tensor<73x74xi1>, tensor<1xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<73x74xi1>, tensor<1x74xi1>) -> tensor<73x74xi1>
    %1 = tosa.sigmoid %arg2 : (tensor<28xf32>) -> tensor<28xf32>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<28xf32>) -> tensor<28xf32>
    %3 = tosa.bitwise_and %0, %0 : (tensor<73x74xi1>, tensor<73x74xi1>) -> tensor<73x74xi1>
    %4 = tosa.greater_equal %2, %1 : (tensor<28xf32>, tensor<28xf32>) -> tensor<28xi1>
    %5 = tosa.reduce_min %4 {axis = 0 : i32} : (tensor<28xi1>) -> tensor<1xi1>
    %6 = tosa.logical_and %5, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %3, %6 : tensor<73x74xi1>, tensor<1xi1>
  }
}
