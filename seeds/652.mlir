module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<40x70xi64>, %arg2: tensor<26xi1>, %arg3: tensor<43x79x40x19x60xf32>) -> (tensor<1xi1>, tensor<43x79x40x19x60xf32>, tensor<70xi1>, tensor<1x1x2xi1>) {
    %0 = tosa.logical_not %arg0 : (tensor<i1>) -> tensor<i1>
    %1 = tosa.argmax %arg1 {axis = 0 : i32} : (tensor<40x70xi64>) -> tensor<70xi32>
    %2 = tosa.reduce_all %arg2 {axis = 0 : i32} : (tensor<26xi1>) -> tensor<1xi1>
    %3 = tosa.log %arg3 : (tensor<43x79x40x19x60xf32>) -> tensor<43x79x40x19x60xf32>
    %r_4 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %0, %r_4 : (tensor<i1>, !tosa.shape<3>) -> tensor<1x1x1xi1>
    %5 = tosa.clz %1 : (tensor<70xi32>) -> tensor<70xi32>
    %6 = tosa.logical_not %4 : (tensor<1x1x1xi1>) -> tensor<1x1x1xi1>
    %7 = tosa.concat %6, %6 {axis = 2 : i32} : (tensor<1x1x1xi1>, tensor<1x1x1xi1>) -> tensor<1x1x2xi1>
    %8 = tosa.logical_and %7, %7 : (tensor<1x1x2xi1>, tensor<1x1x2xi1>) -> tensor<1x1x2xi1>
    %9 = tosa.bitwise_not %8 : (tensor<1x1x2xi1>) -> tensor<1x1x2xi1>
    %10 = tosa.equal %1, %5 : (tensor<70xi32>, tensor<70xi32>) -> tensor<70xi1>
    %11 = tosa.arithmetic_right_shift %9, %7 {round = true} : (tensor<1x1x2xi1>, tensor<1x1x2xi1>) -> tensor<1x1x2xi1>
    return %2, %3, %10, %11 : tensor<1xi1>, tensor<43x79x40x19x60xf32>, tensor<70xi1>, tensor<1x1x2xi1>
  }
}
