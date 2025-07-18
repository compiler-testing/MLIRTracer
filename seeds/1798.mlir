module {
  func.func @main(%arg0: tensor<81x27x23xi1>, %arg1: tensor<81x21x23xi1>, %arg2: tensor<46x15x76x55x48x82xi32>, %arg3: tensor<46x1x1x55x1x82xi32>) -> (tensor<46x15x76x55x48x82xi1>, tensor<81x48x23xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<81x27x23xi1>, tensor<81x21x23xi1>) -> tensor<81x48x23xi1>
    %1 = tosa.equal %arg2, %arg3 : (tensor<46x15x76x55x48x82xi32>, tensor<46x1x1x55x1x82xi32>) -> tensor<46x15x76x55x48x82xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<46x15x76x55x48x82xi1>, tensor<46x15x76x55x48x82xi1>) -> tensor<46x15x76x55x48x82xi1>
    %3 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<81x48x23xi1>, tensor<81x48x23xi1>) -> tensor<81x48x23xi1>
    %4 = tosa.identity %2 : (tensor<46x15x76x55x48x82xi1>) -> tensor<46x15x76x55x48x82xi1>
    %5 = tosa.logical_xor %3, %3 : (tensor<81x48x23xi1>, tensor<81x48x23xi1>) -> tensor<81x48x23xi1>
    %6 = tosa.bitwise_not %5 : (tensor<81x48x23xi1>) -> tensor<81x48x23xi1>
    %7 = tosa.bitwise_xor %6, %3 : (tensor<81x48x23xi1>, tensor<81x48x23xi1>) -> tensor<81x48x23xi1>
    %8 = tosa.logical_or %7, %3 : (tensor<81x48x23xi1>, tensor<81x48x23xi1>) -> tensor<81x48x23xi1>
    return %4, %8 : tensor<46x15x76x55x48x82xi1>, tensor<81x48x23xi1>
  }
}
