module {
  func.func @main(%arg0: tensor<89x6x33x71x49xi64>, %arg1: tensor<1x1x1x1x49xi64>, %arg2: tensor<85x21x4x64xi8>, %arg3: tensor<49x13x64x51x41xf32>, %arg4: tensor<15x20x82x58xi1>) -> (tensor<85x21x1x64xi8>, tensor<49x13x64x51x41xf32>, tensor<15x1x82x58xi1>, tensor<89x6x33x71x49xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<89x6x33x71x49xi64>, tensor<1x1x1x1x49xi64>) -> tensor<89x6x33x71x49xi1>
    %1 = tosa.reduce_min %arg2 {axis = 2 : i32} : (tensor<85x21x4x64xi8>) -> tensor<85x21x1x64xi8>
    %2 = tosa.rsqrt %arg3 : (tensor<49x13x64x51x41xf32>) -> tensor<49x13x64x51x41xf32>
    %3 = tosa.clz %0 : (tensor<89x6x33x71x49xi1>) -> tensor<89x6x33x71x49xi1>
    %4 = tosa.reduce_all %arg4 {axis = 1 : i32} : (tensor<15x20x82x58xi1>) -> tensor<15x1x82x58xi1>
    %5 = tosa.sub %4, %4 : (tensor<15x1x82x58xi1>, tensor<15x1x82x58xi1>) -> tensor<15x1x82x58xi1>
    %6 = tosa.arithmetic_right_shift %3, %0 {round = true} : (tensor<89x6x33x71x49xi1>, tensor<89x6x33x71x49xi1>) -> tensor<89x6x33x71x49xi1>
    return %1, %2, %5, %6 : tensor<85x21x1x64xi8>, tensor<49x13x64x51x41xf32>, tensor<15x1x82x58xi1>, tensor<89x6x33x71x49xi1>
  }
}
