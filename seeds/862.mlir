module {
  func.func @main(%arg0: tensor<12x35x20xi64>, %arg1: tensor<68x14x94x8x82xi1>, %arg2: tensor<1x1x94x8x82xi1>, %arg3: tensor<28x74x63x86x20x87xi32>, %arg4: tensor<28x74x63x86x1x87xi32>) -> (tensor<12x1x20xi64>, tensor<82x8x14x94x68xi1>, tensor<28x74x63x86x20x87xi32>) {
    %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<12x35x20xi64>) -> tensor<12x1x20xi64>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<12x1x20xi64>) -> tensor<12x1x20xi64>
    %2 = tosa.logical_or %arg1, %arg2 : (tensor<68x14x94x8x82xi1>, tensor<1x1x94x8x82xi1>) -> tensor<68x14x94x8x82xi1>
    %3 = tosa.intdiv %arg3, %arg4 : (tensor<28x74x63x86x20x87xi32>, tensor<28x74x63x86x1x87xi32>) -> tensor<28x74x63x86x20x87xi32>
    %4 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %5 = tosa.transpose %2 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<68x14x94x8x82xi1>) -> tensor<82x8x14x94x68xi1>
    %6 = tosa.intdiv %3, %3 : (tensor<28x74x63x86x20x87xi32>, tensor<28x74x63x86x20x87xi32>) -> tensor<28x74x63x86x20x87xi32>
    return %1, %5, %6 : tensor<12x1x20xi64>, tensor<82x8x14x94x68xi1>, tensor<28x74x63x86x20x87xi32>
  }
}
