module {
  func.func @main(%arg0: tensor<91x27x48x97x80x34xi8>, %arg1: tensor<91x27x48x97x1x1xi8>, %arg2: tensor<f32>, %arg3: tensor<6x6x12xi1>, %arg4: tensor<6x6x1xi1>) -> (tensor<91x27x48x97x80x34xi8>, tensor<18x18x36xi1>, tensor<f32>, tensor<6x6x12xi1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<91x27x48x97x80x34xi8>, tensor<91x27x48x97x1x1xi8>) -> tensor<91x27x48x97x80x34xi8>
    %1 = tosa.log %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<91x27x48x97x80x34xi8>, tensor<91x27x48x97x80x34xi8>) -> tensor<91x27x48x97x80x34xi8>
    %3 = tosa.bitwise_not %2 : (tensor<91x27x48x97x80x34xi8>) -> tensor<91x27x48x97x80x34xi8>
    %4 = tosa.logical_or %arg3, %arg4 : (tensor<6x6x12xi1>, tensor<6x6x1xi1>) -> tensor<6x6x12xi1>
    %t_5 = tosa.const_shape {values = dense<[ 3, 3, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.tile %4, %t_5 : (tensor<6x6x12xi1>, !tosa.shape<3>) -> tensor<18x18x36xi1>
    %6 = tosa.logical_or %5, %5 : (tensor<18x18x36xi1>, tensor<18x18x36xi1>) -> tensor<18x18x36xi1>
    %7 = tosa.log %1 : (tensor<f32>) -> tensor<f32>
    %8 = tosa.logical_or %4, %4 : (tensor<6x6x12xi1>, tensor<6x6x12xi1>) -> tensor<6x6x12xi1>
    return %3, %6, %7, %8 : tensor<91x27x48x97x80x34xi8>, tensor<18x18x36xi1>, tensor<f32>, tensor<6x6x12xi1>
  }
}
