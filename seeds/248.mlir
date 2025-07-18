module {
  func.func @main(%arg0: tensor<93x70x77xi32>, %arg1: tensor<93x1x1xi32>, %arg2: tensor<70x99x13xf32>) -> (tensor<93x77x70xi32>, tensor<70x99x13xf32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<93x70x77xi32>, tensor<93x1x1xi32>) -> tensor<93x70x77xi32>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<93x70x77xi32>, tensor<93x70x77xi32>) -> tensor<93x70x77xi32>
    %2 = tosa.ceil %arg2 : (tensor<70x99x13xf32>) -> tensor<70x99x13xf32>
    %3 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %4 = tosa.transpose %1 {perms = array<i32: 0, 2, 1>} : (tensor<93x70x77xi32>) -> tensor<93x77x70xi32>
    %5 = tosa.bitwise_and %4, %4 : (tensor<93x77x70xi32>, tensor<93x77x70xi32>) -> tensor<93x77x70xi32>
    %6 = tosa.reverse %2 {axis = 1 : i32} : (tensor<70x99x13xf32>) -> tensor<70x99x13xf32>
    return %5, %6 : tensor<93x77x70xi32>, tensor<70x99x13xf32>
  }
}
