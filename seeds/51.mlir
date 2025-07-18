module {
  func.func @main(%arg0: tensor<42x72xi32>, %arg1: tensor<97x99x42x21x46x54xf32>) -> (tensor<97x99x42x21x46x54xf32>, tensor<97x99x42x21x46x54xf32>, tensor<42x72xi32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<42x72xi32>) -> tensor<42x72xi32>
    %1 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0, 1>} : (tensor<42x72xi32>) -> tensor<42x72xi32>
    %3 = tosa.floor %arg1 : (tensor<97x99x42x21x46x54xf32>) -> tensor<97x99x42x21x46x54xf32>
    %4 = tosa.logical_right_shift %2, %2 : (tensor<42x72xi32>, tensor<42x72xi32>) -> tensor<42x72xi32>
    %5 = tosa.tanh %3 : (tensor<97x99x42x21x46x54xf32>) -> tensor<97x99x42x21x46x54xf32>
    %6 = tosa.sub %3, %3 : (tensor<97x99x42x21x46x54xf32>, tensor<97x99x42x21x46x54xf32>) -> tensor<97x99x42x21x46x54xf32>
    %7 = tosa.intdiv %0, %4 : (tensor<42x72xi32>, tensor<42x72xi32>) -> tensor<42x72xi32>
    return %5, %6, %7 : tensor<97x99x42x21x46x54xf32>, tensor<97x99x42x21x46x54xf32>, tensor<42x72xi32>
  }
}
