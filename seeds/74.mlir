module {
  func.func @main(%arg0: tensor<17x51x5xi1>, %arg1: tensor<17x51x1xi1>, %arg2: tensor<3x35x95x8x41xi64>, %arg3: tensor<3x35x95x1x41xi64>, %arg4: tensor<61x95x66x28x12x86xf32>) -> (tensor<3x35x95x8x41xi64>, tensor<17x1x5xi1>, tensor<61x95x66x28x12x86xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<17x51x5xi1>, tensor<17x51x1xi1>) -> tensor<17x51x5xi1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<17x51x5xi1>, tensor<17x51x5xi1>) -> tensor<17x51x5xi1>
    %2 = tosa.sub %1, %1 : (tensor<17x51x5xi1>, tensor<17x51x5xi1>) -> tensor<17x51x5xi1>
    %3 = tosa.minimum %arg2, %arg3 : (tensor<3x35x95x8x41xi64>, tensor<3x35x95x1x41xi64>) -> tensor<3x35x95x8x41xi64>
    %4 = tosa.logical_left_shift %3, %3 : (tensor<3x35x95x8x41xi64>, tensor<3x35x95x8x41xi64>) -> tensor<3x35x95x8x41xi64>
    %5 = tosa.concat %2, %1 {axis = 1 : i32} : (tensor<17x51x5xi1>, tensor<17x51x5xi1>) -> tensor<17x102x5xi1>
    %6 = tosa.sigmoid %arg4 : (tensor<61x95x66x28x12x86xf32>) -> tensor<61x95x66x28x12x86xf32>
    %7 = tosa.abs %4 : (tensor<3x35x95x8x41xi64>) -> tensor<3x35x95x8x41xi64>
    %8 = tosa.reduce_sum %5 {axis = 1 : i32} : (tensor<17x102x5xi1>) -> tensor<17x1x5xi1>
    %9 = tosa.rsqrt %6 : (tensor<61x95x66x28x12x86xf32>) -> tensor<61x95x66x28x12x86xf32>
    return %7, %8, %9 : tensor<3x35x95x8x41xi64>, tensor<17x1x5xi1>, tensor<61x95x66x28x12x86xf32>
  }
}
