module {
  func.func @main(%arg0: tensor<39x95x57x67xi16>, %arg1: tensor<1x95x1x1xi16>, %arg2: tensor<36x81x37xf32>) -> (tensor<39x95x57x67xi16>, tensor<2x2x37xi1>, tensor<36x81x37xi1>, tensor<1x1xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<39x95x57x67xi16>, tensor<1x95x1x1xi16>) -> tensor<39x95x57x67xi16>
    %1 = tosa.tanh %arg2 : (tensor<36x81x37xf32>) -> tensor<36x81x37xf32>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<39x95x57x67xi16>, tensor<39x95x57x67xi16>) -> tensor<39x95x57x67xi16>
    %3 = tosa.equal %1, %1 : (tensor<36x81x37xf32>, tensor<36x81x37xf32>) -> tensor<36x81x37xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<36x81x37xi1>, tensor<36x81x37xi1>) -> tensor<36x81x37xi1>
    %5 = tosa.reduce_any %4 {axis = 1 : i32} : (tensor<36x81x37xi1>) -> tensor<36x1x37xi1>
    %6 = tosa.reduce_product %5 {axis = 0 : i32} : (tensor<36x1x37xi1>) -> tensor<1x1x37xi1>
    %7 = tosa.logical_left_shift %6, %6 : (tensor<1x1x37xi1>, tensor<1x1x37xi1>) -> tensor<1x1x37xi1>
    %t_8 = tosa.const_shape {values = dense<[ 2, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %8 = tosa.tile %7, %t_8 : (tensor<1x1x37xi1>, !tosa.shape<3>) -> tensor<2x2x37xi1>
    %9 = tosa.logical_right_shift %8, %8 : (tensor<2x2x37xi1>, tensor<2x2x37xi1>) -> tensor<2x2x37xi1>
    %10 = tosa.sigmoid %1 : (tensor<36x81x37xf32>) -> tensor<36x81x37xf32>
    %11 = tosa.floor %10 : (tensor<36x81x37xf32>) -> tensor<36x81x37xf32>
    %12 = tosa.argmax %1 {axis = 0 : i32} : (tensor<36x81x37xf32>) -> tensor<81x37xi32>
    %13 = tosa.abs %11 : (tensor<36x81x37xf32>) -> tensor<36x81x37xf32>
    %14 = tosa.log %13 : (tensor<36x81x37xf32>) -> tensor<36x81x37xf32>
    %15 = tosa.minimum %14, %13 : (tensor<36x81x37xf32>, tensor<36x81x37xf32>) -> tensor<36x81x37xf32>
    %16 = tosa.equal %15, %14 : (tensor<36x81x37xf32>, tensor<36x81x37xf32>) -> tensor<36x81x37xi1>
    %17 = tosa.greater %12, %12 : (tensor<81x37xi32>, tensor<81x37xi32>) -> tensor<81x37xi1>
    %18 = tosa.reduce_min %17 {axis = 0 : i32} : (tensor<81x37xi1>) -> tensor<1x37xi1>
    %19 = tosa.reduce_all %18 {axis = 1 : i32} : (tensor<1x37xi1>) -> tensor<1x1xi1>
    return %2, %9, %16, %19 : tensor<39x95x57x67xi16>, tensor<2x2x37xi1>, tensor<36x81x37xi1>, tensor<1x1xi1>
  }
}
