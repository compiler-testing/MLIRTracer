module {
  func.func @main(%arg0: tensor<16x7x56x64xi1>, %arg1: tensor<42x30x13x74xi8>, %arg2: tensor<42x30x1x1xi8>, %arg3: tensor<67x75xf32>) -> (tensor<42x30x13x74xi1>, tensor<401408xi1>, tensor<1x75xf32>, tensor<67x1xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 401408 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<16x7x56x64xi1>, !tosa.shape<1>) -> tensor<401408xi1>
    %1 = tosa.greater %arg1, %arg2 : (tensor<42x30x13x74xi8>, tensor<42x30x1x1xi8>) -> tensor<42x30x13x74xi1>
    %2 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<401408xi1>, tensor<401408xi1>) -> tensor<401408xi1>
    %3 = tosa.logical_and %2, %0 : (tensor<401408xi1>, tensor<401408xi1>) -> tensor<401408xi1>
    %4 = tosa.arithmetic_right_shift %3, %0 {round = true} : (tensor<401408xi1>, tensor<401408xi1>) -> tensor<401408xi1>
    %5 = tosa.sigmoid %arg3 : (tensor<67x75xf32>) -> tensor<67x75xf32>
    %6 = tosa.equal %5, %5 : (tensor<67x75xf32>, tensor<67x75xf32>) -> tensor<67x75xi1>
    %7 = tosa.reduce_product %5 {axis = 0 : i32} : (tensor<67x75xf32>) -> tensor<1x75xf32>
    %8 = tosa.reduce_product %6 {axis = 1 : i32} : (tensor<67x75xi1>) -> tensor<67x1xi1>
    return %1, %4, %7, %8 : tensor<42x30x13x74xi1>, tensor<401408xi1>, tensor<1x75xf32>, tensor<67x1xi1>
  }
}
