module {
  func.func @main(%arg0: tensor<27x92x39x78xi64>, %arg1: tensor<80x99x60xf32>) -> (tensor<160x198x120xf32>, tensor<1x99x60xf32>, tensor<i32>) {
    %r_0 = tosa.const_shape {values = dense<[ 7556328 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<27x92x39x78xi64>, !tosa.shape<1>) -> tensor<7556328xi64>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<7556328xi64>, tensor<7556328xi64>) -> tensor<15112656xi64>
    %2 = tosa.greater %1, %1 : (tensor<15112656xi64>, tensor<15112656xi64>) -> tensor<15112656xi1>
    %3 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<15112656xi1>) -> tensor<1xi1>
    %4 = tosa.clz %3 : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.bitwise_not %4 : (tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.exp %arg1 : (tensor<80x99x60xf32>) -> tensor<80x99x60xf32>
    %7 = tosa.argmax %5 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %t_8 = tosa.const_shape {values = dense<[ 2, 2, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %8 = tosa.tile %6, %t_8 : (tensor<80x99x60xf32>, !tosa.shape<3>) -> tensor<160x198x120xf32>
    %9 = tosa.reduce_min %6 {axis = 0 : i32} : (tensor<80x99x60xf32>) -> tensor<1x99x60xf32>
    %10 = tosa.bitwise_xor %7, %7 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %8, %9, %10 : tensor<160x198x120xf32>, tensor<1x99x60xf32>, tensor<i32>
  }
}
