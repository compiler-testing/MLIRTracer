module {
  func.func @main(%arg0: tensor<82xi8>, %arg1: tensor<1xi8>, %arg2: tensor<f32>, %arg3: tensor<77x14x62x34xi32>, %arg4: tensor<1x1x1x34xi32>) -> (tensor<f32>, tensor<82xi8>, tensor<77x14x62x34xi32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<82xi8>, tensor<1xi8>) -> tensor<82xi8>
    %1 = tosa.clamp %0 {min_val = -3 : i8, max_val = 95 : i8} : (tensor<82xi8>) -> tensor<82xi8>
    %2 = tosa.identity %1 : (tensor<82xi8>) -> tensor<82xi8>
    %3 = tosa.arithmetic_right_shift %2, %0 {round = true} : (tensor<82xi8>, tensor<82xi8>) -> tensor<82xi8>
    %4 = tosa.sub %3, %2 : (tensor<82xi8>, tensor<82xi8>) -> tensor<82xi8>
    %5 = tosa.logical_left_shift %4, %2 : (tensor<82xi8>, tensor<82xi8>) -> tensor<82xi8>
    %6 = tosa.reverse %5 {axis = 0 : i32} : (tensor<82xi8>) -> tensor<82xi8>
    %7 = tosa.add %6, %2 : (tensor<82xi8>, tensor<82xi8>) -> tensor<82xi8>
    %8 = tosa.logical_left_shift %7, %4 : (tensor<82xi8>, tensor<82xi8>) -> tensor<82xi8>
    %9 = tosa.bitwise_and %8, %3 : (tensor<82xi8>, tensor<82xi8>) -> tensor<82xi8>
    %10 = tosa.bitwise_and %9, %8 : (tensor<82xi8>, tensor<82xi8>) -> tensor<82xi8>
    %11 = tosa.maximum %10, %8 : (tensor<82xi8>, tensor<82xi8>) -> tensor<82xi8>
    %12 = tosa.exp %arg2 : (tensor<f32>) -> tensor<f32>
    %13 = tosa.abs %11 : (tensor<82xi8>) -> tensor<82xi8>
    %14 = tosa.intdiv %arg3, %arg4 : (tensor<77x14x62x34xi32>, tensor<1x1x1x34xi32>) -> tensor<77x14x62x34xi32>
    return %12, %13, %14 : tensor<f32>, tensor<82xi8>, tensor<77x14x62x34xi32>
  }
}
