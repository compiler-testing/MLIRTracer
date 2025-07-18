module {
  func.func @main(%arg0: tensor<2x36xi1>, %arg1: tensor<2x1xi1>, %arg2: tensor<68x2x4x60xi32>, %arg3: tensor<68x1x1x1xi32>, %arg4: tensor<45xf32>) -> (tensor<68x2x4x60xi32>, tensor<1x36xi1>, tensor<1xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<2x36xi1>, tensor<2x1xi1>) -> tensor<2x36xi1>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<68x2x4x60xi32>, tensor<68x1x1x1xi32>) -> tensor<68x2x4x60xi32>
    %2 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = tosa.transpose %0 {perms = array<i32: 0, 1>} : (tensor<2x36xi1>) -> tensor<2x36xi1>
    %4 = tosa.logical_right_shift %3, %0 : (tensor<2x36xi1>, tensor<2x36xi1>) -> tensor<2x36xi1>
    %5 = tosa.reduce_any %4 {axis = 0 : i32} : (tensor<2x36xi1>) -> tensor<1x36xi1>
    %6 = tosa.rsqrt %arg4 : (tensor<45xf32>) -> tensor<45xf32>
    %7 = tosa.reduce_min %6 {axis = 0 : i32} : (tensor<45xf32>) -> tensor<1xf32>
    return %1, %5, %7 : tensor<68x2x4x60xi32>, tensor<1x36xi1>, tensor<1xf32>
  }
}
